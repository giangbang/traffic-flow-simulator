import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np

class agents():
    def __init__(self, inchannels, outchannels):
        self.nn = neuralNet(inchannels, outchannels)
        self.actionSize = outchannels
        self.sateSize = inchannels
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('device available: ', self.device)
        self.mem = replayMemory()
        self.nn.to(self.device)
        self.batchThreshold = int(120)
        self.discount = 0.
        self.optim = torch.optim.Adam(self.nn.parameters(), lr = 1e-2)
        self.lossfunc = nn.MSELoss()
        self.loss_his = []
        
    def select_actions(self, epsilon, state, out=False):
        rand = random.random()
        res = None
        if rand <= epsilon:
            with torch.no_grad():
                self.nn.eval()
                state = torch.from_numpy(state.astype(np.float32)).to(self.device)
                res = self.nn(state)
                if (out):
                    print(res)
                res = res.argmax(dim=1).squeeze().detach().cpu()
                res = np.array([res])
        else:
            res = np.random.randint(self.actionSize, size=1)
        
        return res[np.newaxis, ...]
            
    def add_memmory(self, state, action, nextState, reward):
        self.mem.push(state, action, nextState, reward)
        
    def train(self):
        if len(self.mem) < self.batchThreshold:
            return
        
        batch = self.mem.sample()
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
        
        self.nn.train()
        loss = self.__loss__(state, action, nextState, reward)
        self.optim.zero_grad()
        self.loss_his.append(loss.item())
        loss.backward()
        self.optim.step()
        
    def save_policy(self, path):
        torch.save(self.nn.state_dict(), path)
        
    def load_policy(self, path):
        self.nn.load_state_dict(torch.load(path))
        
    def plot(self):
        x = range(len(self.loss_his))
        
        plt.plot(x, self.loss_his, c='r')
        plt.show()
        
    def __loss__(self, state, action, nextState, reward):
    
        value = self.nn(state)
       
        predictedReward = torch.gather(value, 1, action)
        nextPredictedReward = self.nn(nextState)
        estimatedReward = reward + self.discount * torch.max(nextPredictedReward, dim=1)[0]
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
    def __init__(self):
        self.maxSize = int(1e3)
        self.mem = []
        self.pos = int(0)
        self.batch = 32
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
        