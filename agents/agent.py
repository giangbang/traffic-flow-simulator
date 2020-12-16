import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np

class agents():
    def __init__(self, inchannels, outchannels):
        self.memory = replayMemory(batch_size=4)
        self.action_size = outchannels
        self.sate_size = inchannels
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('device available: ', self.device)
        self.batchThreshold = int(5)
        self.discount = 1
        self.loss_his = []
        
    def select_action(self, epsilon, state):
        pass
        
    def add_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        
        
    def train(self):
        pass
        
    def save(self, path):
        pass
        
    def load(self, path):
        pass
     
     
class tls_based_agent(agents):
    def __init__(self, inchannels, outchannels):
        super().__init__(inchannels, outchannels)
        self.nn = neuralNet(inchannels, outchannels)
        self.nn.to(self.device)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr = 1e-2)
        self.lossfunc = nn.MSELoss()
        
        
    def select_action(self, epsilon, state, out=False):
        rand = random.random()
        res = None
        if rand <= epsilon:
            with torch.no_grad():
                self.nn.eval()
                state = torch.from_numpy(state.astype(np.float32)).to(self.device)
                # print(state)
                res = self.nn(state)
                # print(res)
                if (out):
                    print(res)
                res = res.argmax(dim=1).squeeze().detach().cpu()
                res = np.array([res])
        else:
            res = np.random.randint(self.action_size, size=1)
        # print(res)
        return res[np.newaxis, ...]
            
    
    def train(self, log=False):
        if len(self.memory) < self.batchThreshold:
            return
        
        batch = self.memory.sample()
        state, action, nextState, reward = None, None, None, None
        init = True
        for instance in batch:
            if init:
                init = False
                state = instance[0]
                action = instance[1]
                nextState = instance[2]
                reward = instance[3]
            else:
               
                state = np.concatenate((state, instance[0]), axis=0)
                action = np.concatenate((action, instance[1]), axis=0)
                nextState = np.concatenate((nextState, instance[2]), axis=0)
                reward = np.concatenate((reward, instance[3]), axis=0)
            
        state = torch.from_numpy(np.array(state).astype(np.float32)).to(self.device)
        action = torch.from_numpy(np.array(action).astype(np.int64)).to(self.device)
        nextState = torch.from_numpy(np.array(nextState).astype(np.float32)).to(self.device)
        reward = torch.from_numpy(np.array(reward).astype(np.float32)).to(self.device)
        
        if log:
            print('-'*40)
            print('state')
            print(state)
            print('action')
            print(action)
            print('next_state')
            print(nextState)
            print('reward')
            print(reward)
        
        
        self.nn.train()
        loss = self.__loss__(state, action, nextState, reward, log)
        self.optim.zero_grad()
        self.loss_his.append(loss.item())
        loss.backward()
        self.optim.step()
        
    def save(self, path):
        torch.save(self.nn.state_dict(), path)
        
    def load(self, path):
        self.nn.load_state_dict(torch.load(path))
        
    def plot(self, name):
        x = np.arange(len(self.loss_his))
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, self.loss_his, label='loss', color='blue', lw=2)
        plt.xlabel('step', fontsize=13)
        plt.ylabel('loss',fontsize=13)
        fig.savefig('./resource/output/1/'+name)
        
    def __loss__(self, state, action, nextState, reward, log):
    
        value = self.nn(state)
        predictedReward = torch.gather(value, 1, action)
        
        nextPredictedReward = self.nn(nextState)
 
        q_prime = torch.max(nextPredictedReward, dim=1, keepdim=True)[0]
        estimatedReward = reward + self.discount * q_prime
        
        if log:
            print('value')
            print(value)
            print('chosen action')
            print(predictedReward)
            print('next steps estimates')
            print(nextPredictedReward)
            print('next steps estimated reward')
            print('this should be the same size as chosen action for computing loss')
            print(estimatedReward)
            print('-'*40)
            
        return self.lossfunc(predictedReward, estimatedReward)
    
class neuralNet(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(neuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inchannels, outchannels, bias=True),
            nn.ReLU(),
            nn.Linear(outchannels, outchannels, bias=True),
        )
        
    def forward(self, x):
        return self.layers(x)
        
class replayMemory:
    def __init__(self, max_size=int(1e3), batch_size=32):
        self.maxSize = int(max_size)
        self.mem = []
        self.pos = int(0)
        self.batch = batch_size
        self.cur = 0
        
    def push(self, state, action, nextState, reward):
        if (self.cur <= self.maxSize):
            self.mem.append((state, action, nextState, reward))
            self.cur += 1
        else:
            self.mem[self.pos] = (state, action, nextState, reward)
            self.pos = random.randint(0, self.maxSize-1)
            
    def sample(self):
        return random.sample(self.mem, self.batch)
        
    def __len__(self):
        return len(self.mem)
        
class agent_maneger():
    def __init__(self, inchannels_dict, outchannels_dict):
        self._agent_list = {}
        self._num_agent = 0
        for a in inchannels_dict:
            self._num_agent += 1 
            self._agent_list[a] = tls_based_agent(inchannels_dict[a], outchannels_dict[a])
        
    def select_action(self, epsilon, state_dict):
        actions = {}
        for a in self._agent_list:
            actions[a] = self._agent_list[a].select_action(epsilon, state_dict[a])
        return actions
        
    def add_memory(self, state, action, next_state, reward, update_res):
        for a in self._agent_list:
            if update_res[a]:
                self._agent_list[a].add_memory(state[a], action[a], next_state[a], reward[a])
            
    def train(self):
        for a in self._agent_list:
            self._agent_list[a].train()
       
    def save(self):
        for a in self._agent_list:
            path = str('./models/'+a+'.pt')
            try:
                self._agent_list[a].save(path)
                print('Done saving', a)
            except:
                print('Error, cannot save '+a+' to '+path)
                
    def plot(self):
        for a in self._agent_list:
            self._agent_list[a].plot(a)
        plt.close("all")
                
    def load(self):
        for a in self._agent_list:
            try:
                self._agent_list[a].load('./models/'+a+'.pt')
                print(a+' model loaded successfully')
            except:
                print('Error: cannot load ' + a)