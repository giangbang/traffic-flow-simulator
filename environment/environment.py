import traci
import numpy as np
import copy
from sumolib import checkBinary  # Checks for the binary in environ vars

sumoBinaryNoGui = checkBinary('sumo')
sumoBinary = checkBinary('sumo-gui')

nogui = [sumoBinaryNoGui, "-c", './network/tmp.sumocfg', '--no-warnings']
gui = [sumoBinary, "-c", './network/tmp.sumocfg', '--quit-on-end']


class environment():
    def __init__(self, options):
        self.options = options
        self._step = 0
        self._total_reward = 0
        self._max_reward = -1e9
        self._min_reward = 1e9
        
    def start(self):
        self._step = 0
        self._total_reward = 0
        self._max_reward = -1e9
        self._min_reward = 1e9
        if (self.options.nogui):
            traci.start(nogui)
        else: 
            traci.start(gui)
        
    def step(self):
        self._step += 1
        traci.simulationStep()
        
    def get_step(self):
        return self._step
        
    def reward(self):
        raise NotImplementedError
           
    def get_state(self):
        raise NotImplementedError
        
    def state_size(self):
        raise NotImplementedError
        
    def action_size(self):
        raise NotImplementedError
        
    def end(self):
        traci.close()
        
    def done(self):
        return traci.simulation.getMinExpectedNumber() <= 0
    
    def do_action(self, action_list):
        raise NotImplementedError
            
# class edge_based(environment):
    # pass


class tls_based_env(environment):
    def __init__(self, options):
        super().__init__(options)
        traci.start(nogui)
        self._tls_list = list(traci.trafficlight.getIDList())
        self._phase_dict = {}   # all phases go here
        self._lanes_dict = {}   # all lanes go here
        self._edges_dict = {}   # all edges go here
        self._traffic_timer = {} # timing controllers for all traffic lights 
        self._prev_waiting_time = {}
        self._action_size = None
        self._state_size = None
        
        for i, tls in enumerate(self._tls_list):
            phases = (traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
            lanes = set(traci.trafficlight.getControlledLanes(tls))
            edges = set()
            for lane in lanes:
                edges.add(lane.split('_')[0])
            self._lanes_dict[tls] = list(lanes)
            self._edges_dict[tls] = list(edges)
            self._phase_dict[tls] = list(p.state for p in phases)
            self._prev_waiting_time[tls] = 0
            self._traffic_timer[tls] = clocks()
            
        self.state_size()
        self.action_size()
        traci.close()
        self._waiting_time = copy.deepcopy(self._prev_waiting_time)
        
    def get_state(self):
        state_dict = {}
        for tls in self._tls_list:
            state = []
            for e in self._edges_dict[tls]:
                st = traci.edge.getLastStepHaltingNumber(e)
                st /= (traci.edge.getLastStepVehicleNumber(e)+.01)
                state.append(st)
                st = traci.edge.getLastStepOccupancy(e)
                state.append(st)
            state_dict[tls] = np.array([state]).astype(np.float32)
        
        return state_dict
        
    def state_size(self):
        if self._state_size is None:
            tmp = self.get_state()
            for it in tmp:
                tmp[it] = len(tmp[it].flatten())
            self._state_size = tmp
        return self._state_size
        
    def step(self):
        super().step()
      
        # accumulate reward in each step
        tmp = {}
        for tls in self._tls_list:
            waiting = 0
            for e in self._edges_dict[tls]:
                waiting += traci.edge.getWaitingTime(e)
            tmp[tls] = waiting
            # set the phase of tls as they are
            traci.trafficlight.setPhase(tls, self._traffic_timer[tls].phaseId)
            
        for tls in tmp:
            self._waiting_time[tls] += min(0, self._prev_waiting_time[tls] - tmp[tls])
        self._prev_waiting_time = tmp 
        return self.get_step()
    
    def action_size(self):
        if self._action_size is None:
            tmp = copy.deepcopy(self._phase_dict)
            for it in tmp:
                tmp[it] = int(len(tmp[it])/2)
            self._action_size = tmp
        return self._action_size
        
    def waiting_time_each_vehicle(self):
        tmp = {}
        car_list = list(traci.vehicle.getIDList())
        for car in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car)
        return wait_time   
        
    def waiting_time(self): 
        return sum(self.waiting_time_each_vehicle().values())
        
    def get_tls(self):
        return self._tls_list
        
    def reward(self):
        res = copy.deepcopy(self._waiting_time)
        self.__reset_waiting_time__()
        for tls in res:
            res[tls] = np.array(res[tls]).reshape(1, 1).astype(np.float32)/1000
        total_reward_this_step = sum(res.values())
        self._total_reward += total_reward_this_step
        self._min_reward = min(self._min_reward, total_reward_this_step)
        self._max_reward = max(self._max_reward, total_reward_this_step)
        return res
        
    def cumulative_total_reward(self):
        return self._total_reward
     
    def max_reward(self):
        return self._max_reward
        
    def min_reward(self):
        return self._min_reward
        
    def do_action(self, action_list):
        now = self.get_step()
        res = {}
        for tls in self._tls_list:
            res[tls] = False
            # apply yellow phase if neccessary
            if self._traffic_timer[tls].is_green():
                if (now - self._traffic_timer[tls].start <= self.options.green_phase):
                    continue
                action = action_list[tls]*2
                if (action != self._traffic_timer[tls].phaseId):
                    prev = self._traffic_timer[tls].phaseId
                    self._traffic_timer[tls].set((prev+1)%(self.action_size()[tls]), now)
                    self._traffic_timer[tls].query(action)
                else: 
                    self._traffic_timer[tls].switch(now)
                res[tls] = True
            else: 
                if (now - self._traffic_timer[tls].start <= self.options.yellow_phase):
                    continue
                self._traffic_timer[tls].switch(now)
                res[tls] = True
                
        return res
        
    def __reset_waiting_time__(self):
        for tls in self._tls_list:
            self._waiting_time[tls] = 0
         
class clocks():
    def __init__(self, phaseId=0, start=-1000):
        self.phaseId = phaseId
        self.queried_phase = 0
        self.start = start
        
    def is_green(self):
        return self.phaseId % 2 == 0
        
    def is_yellow(self):
        return not self.is_green()
        
    def set(self, phase, now):
        self.phaseId = phase
        self.start = now
        
    def switch(self, now):
        self.phaseId = self.queried_phase
        self.start = now
  
    def query(self, phase):
        self.queried_phase = phase