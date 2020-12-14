import traci
import numpy as np

class environment():
    def __init__(self):
    
        self.junctions = {}
        self.programLightDict = {}
        self.prev_waiting_time = 0
        self.waiting_time = 0
        trafficlights = list(traci.trafficlight.getIDList())
        for trafficlight in trafficlights:
            lanes = set(traci.trafficlight.getControlledLanes(trafficlight))
            self.junctions[trafficlight] = lanes
            tmp = list(traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight)[0].phases)
            self.programLightDict[trafficlight] = [phase.state for phase in tmp]
        print(self.programLightDict)
        print(self.junctions)
        
    def getWaitingTime(self):
        res = 0
        for i, junc in enumerate(self.junctions): 
            for lane in self.junctions[junc]:
                res += (traci.lane.getWaitingTime(lane))
        return res
        
    def cumulateWaitingTime(self):
        now = self.getWaitingTime()
        # print(no)
        self.waiting_time += max(0, now - self.prev_waiting_time)
        self.prev_waiting_time = now
        
    def reward(self):
        res = -np.array([self.waiting_time]).reshape(1, 1)
        self.waiting_time = 0
        return res/1000
   
     
    def getPhase(self):
        return (traci.trafficlight.getPhase('0'))
        
    def getState2(self):
        state = np.zeros(0)
        for junc in self.junctions:
            numLanes = len(self.junctions[junc])
            numPhases = len(self.programLightDict[junc])
            state = np.zeros((numLanes*2 + numPhases), dtype=np.float32)
            for i, id in enumerate(self.junctions[junc]):
                state[i*2] = traci.lane.getLastStepOccupancy(id)
                state[i*2+1] = traci.lane.getLastStepMeanSpeed(id)
        phase = self.getPhase()
        state[numLanes*2 + phase] = 1
        return state[np.newaxis, ...]
           
    def getState(self):
        state = np.zeros(0)
        for junc in self.junctions:
            numLanes = len(self.junctions[junc])
            numPhases = len(self.programLightDict[junc])
            state = np.zeros((numLanes), dtype=np.float32)
            for i, id in enumerate(self.junctions[junc]):
                state[i] = traci.lane.getLastStepOccupancy(id)
       
        return state[np.newaxis, ...]
        
    def state_size(self):
        return self.getState().flatten().shape[0]
        
    def action_size(self):
        return len(self.programLightDict['0'])
    
    def do_action(self, phaseid):
        for i, junc in enumerate(self.junctions):
            traci.trafficlight.setPhase(junc, phaseid[i])
            