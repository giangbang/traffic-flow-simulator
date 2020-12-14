#!/usr/bin/env python

import os
import sys
import optparse
import time
from environment.environment import tls_based
# from agents.agent import agents
# from agent import agent

# we need to import some python modules from the $SUMO_HOME/tools directory
# os.environ['SUMO_HOME'] = "/usr/share/sumo/"
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('added SUMO_HOME to tools directory')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=True, help="run the commandline version of sumo")
    opt_parser.add_option("--af", action="store_true",
                         default=False, help="run the commandline version of sumo")
    opt_parser.add_option("--episode", type=int, dest='episode',
                         default=10, help="parameters for the commandline training of agent")
    options, args = opt_parser.parse_args()
    return options, args


# contains TraCI control loop
def run(options, args):
    
                             
    env = tls_based(options, args)
    env.start(False)
    print(env.state_size())
    print(env.action_size())
    while not(env.done()):
        env.step()
        if env.get_step()% 100 == 0:
            print(env.get_state())
            
    
    env.end()

    # while episode > 0:
        # episode -= 1
        # traci.start([sumoBinary, "-c", "test.sumocfg",
                             # "--tripinfo-output", "tripinfo.xml"])
        # while traci.simulation.getMinExpectedNumber() > 0:
            # traci.simulationStep()
            # print(env.state_size())
            # print(env.action_size())
            # print(env.reward())
            # break
  
    # traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options, args = get_options()
    run(options, args)
    
    
    # check binary
    # if options.nogui:
        # sumoBinary = checkBinary('sumo')
    # else:
        # sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    # [sumoBinary, "-c", "test.sumocfg",
                             # "--tripinfo-output", "tripinfo.xml"])
                             