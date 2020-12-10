#!/usr/bin/env python

import os
import sys
import optparse
import time
from environment import environment
# from agent import agent

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('added SUMO_HOME to tools directory')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    opt_parser.add_option("--train", action="store_true",
                         default=False, help="run the commandline training of agent")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def run(env, episode):
    options = get_options()
    
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "test.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
                             
    env = environment()
    # agt = agent()
    

    while episode > 0:
        episode -= 1
        traci.start([sumoBinary, "-c", "test.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            print(env.state_size())
            print(env.action_size())
            print(env.reward())
            break
  
    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    run(1)