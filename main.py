#!/usr/bin/env python

import os
import sys
import optparse
import time
from environment.environment import *
from agents.agent import *


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
                         default=False, help="run the commandline version of sumo")
    opt_parser.add_option("--train", action="store_true",
                         default=False, help="run the commandline version of sumo")
    opt_parser.add_option("--episode", type=int, dest='episode',
                         default=10, help="parameters for the commandline training of agent")
    opt_parser.add_option("--yphs", type=int, dest='yellow_phase',
                         default=10, help="parameters for the commandline controlling of agent")
    opt_parser.add_option("--gphs", type=int, dest='green_phase',
                         default=20, help="parameters for the commandline controlling of agent")
    opt_parser.add_option("--stop", action="store_true",
                         default=False, help="searly stop, just for debuging")
    options, args = opt_parser.parse_args()
    return options, args


# contains TraCI control loop
def run(options):
    print(options)
    env = tls_based_env(options)
    agt = agent_maneger(env.state_size(), env.action_size())
    agt.load()
    print(env.state_size())
    print(env.action_size())
    for eps in range(options.episode):
        print('='*40)
        print('Episode', str(eps))
        epsilon = ( (eps) / (options.episode+1) )
        if not options.train:
          epsilon = 1
        # epsilon=0
        print('epsilon = '+str(epsilon))
        # simulation start
        env.start()
        
        state = env.get_state()
        action = agt.select_action(epsilon, state)
        # some action might not be applied, so here we just update 
        # the agent's memory when its action is considered
        update_res = env.do_action(action) 
        
        # simulation loop
        while not(env.done()) and env.get_step() < 5000:
          env.step()
          
          if options.train:
            agt.train()
          if env.get_step() % 20 == 0:
            reward = env.reward(update_res)
            next_state = env.get_state()
            agt.add_memory(state, action, next_state, reward, update_res)
            action = agt.select_action(0, next_state)
            state = next_state
            update_res = env.do_action(action)
              
          if env.get_step() >= 100 and options.stop:
              break
        env.end()
        
        if options.train:
          for e in range(500):
            agt.train()
            
        if (env.get_step() >= 5000):
          print("cannot finish the episode")
        print('total reward:', str(env.cumulative_total_reward()))
        print('minimum reward:', str(env.min_reward()))
        print('maximum reward:', str(env.max_reward()))
        print('step: ', str(env.get_step()))
    agt.save()
    sys.stdout.flush()
    if (options.train):
      agt.plot()


# main entry point
if __name__ == "__main__":
    options, _ = get_options()
    run(options)
    