import traci
import numpy as np
import copy
from sumolib import checkBinary  # Checks for the binary in environ vars

sumoBinaryNoGui = checkBinary('sumo')
sumoBinary = checkBinary('sumo-gui')

nogui = [sumoBinaryNoGui, "-c", './network/tmp.sumocfg', '--no-warnings']
gui = [sumoBinary, "-c", './network/tmp.sumocfg', '--quit-on-end']


class environment():
    def __init__(self, options, args):
        self.args = args
        self.options = options
        self._step = 0
        
    def start(self, noGui=True):
        if (noGui):
            traci.start(nogui)
        else: 
            traci.start(gui)
        
    def step(self, train=True):
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
    
    def do_action(self, phaseid):
        raise NotImplementedError
            
# class edge_based(environment):
    # pass


class tls_based(environment):
    def __init__(self, options, args):
        super().__init__(options, args)
        traci.start(nogui)
        self._tls_list = list(traci.trafficlight.getIDList())
        self._phase_dict = {}
        self._lanes_dict = {}
        self._edges_dict = {}
        self._prev_waiting_time = {}
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
            state_dict[tls] = np.array(state).astype(np.float32)
        return state_dict
        
    def state_size(self):
        tmp = self.get_state()
        for it in tmp:
            tmp[it] = len(tmp[it])
        return tmp
        
    def step(self, train=True):
        super().step()
        # accumulate reward in each step
        tmp = {}
        for tls in self._tls_list:
            waiting = 0
            for e in self._edges_dict[tls]:
                waiting += traci.edge.getWaitingTime(e)
            tmp[tls] = waiting
            
        for tls in tmp:
            self._waiting_time[tls] += min(0, self._prev_waiting_time[tls] - tmp[tls])
        self._prev_waiting_time = tmp 
        return self.get_step()
    
    def action_size(self):
        tmp = copy.deepcopy(self._phase_dict)
        for it in tmp:
            tmp[it] = int(len(tmp[it])/2)
        return tmp
        
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
        return res
        
    def __reset_waiting_time__(self):
        for tls in self._tls_list:
            self._waiting_time[tls] = 0