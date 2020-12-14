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
    
    def do_action(self, phaseid):
        for i, junc in enumerate(traci.trafficlight.getIDList()):
            traci.trafficlight.setPhase(junc, phaseid[i])
            
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
        for i, tls in enumerate(self._tls_list):
            phases = (traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
            lanes = set(traci.trafficlight.getControlledLanes(tls))
            edges = set()
            for lane in lanes:
                edges.add(lane.split('_')[0])
            self._lanes_dict[tls] = list(lanes)
            self._edges_dict[tls] = list(edges)
            self._phase_dict[tls] = list(p.state for p in phases)
        traci.close()
        
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
    
    def action_size(self):
        tmp = copy.deepcopy(self._phase_dict)
        for it in tmp:
            tmp[it] = int(len(tmp[it])/2)
        return tmp