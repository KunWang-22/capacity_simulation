
import argparse
from distutils.command.build import build
from itertools import count

import os, sys, random
import numpy as np
import pandas as pd
from environment import HVAC_env
from building import HVAC_Building
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from matplotlib import pyplot as plt
import math


parser = argparse.ArgumentParser()
#当前的模式、是否需要下载训练好的agent
parser.add_argument('--mode', default='generate', type=str) # mode = 'train' or 'test'
parser.add_argument('--test_iteration', default=5, type=int)
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=100000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
# optional parameters
#parser.add_argument('--render', default=True, type=bool) # show UI or not
parser.add_argument('--save_interval', default=50, type=int) # how much episodes to save the agent
parser.add_argument('--exploration_noise', default=0.32, type=float)  
parser.add_argument('--max_episode', default=8000, type=int) # num of games
parser.add_argument('--update_iteration', default=20, type=int)
#是否需要确定随机种子，保证能复现
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=32, type=int)

# new parameter
parser.add_argument('--building_type', type=str)



args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

#设置随机种子
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

my_buliding = HVAC_Building('office', random.randrange(3000,4001), 4, random.randrange(60,100) , 16)
# my_buliding = HVAC_Building('commercial', random.randrange(8000,16001), 6, random.randrange(12,24), 16)
# my_buliding = HVAC_Building('hotel', random.randrange(3000,4001), 6, random.randrange(30,100), 16)

env = HVAC_env(args.random_seed, my_buliding)
#state/action维度等信息
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.action_max
#min_Val = torch.tensor(1e-7).float().to(device) # min value
#运行存储的文件夹名字

directory = '../model/' + args.building_type + '/' + str(args.random_seed) + '/'
if not os.path.exists(directory):
    os.makedirs(directory)


#存储记忆
class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size  #最大记忆容量
        self.ptr = 0  # pointer

    def store(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)   # 'storage' is a List; 'data' is a Tuple

    def sample(self, batch_size):
        #选取一定量的样本编号
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        #根据编号把样本数据对应取出来
        x, y, u, r, d = [], [], [], [], [] # all are 'list
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))  # Why copy = false ?
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
        # turn 'list' into 'array' , reshape(-1,1) is not necessary

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()       
        L1 = 256
        L2 = 128
        self.l1 = nn.Linear(state_dim, L1)
        self.l2 = nn.Linear(L1, L2)
        self.l3 = nn.Linear(L2, action_dim)
        
        self.max_action = max_action #torch.from_numpy(max_action)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * (torch.tanh(self.l3(x))+1)/2 # tanh(x) is between (-1,1)
        #x = torch.tensor(x, dtype=torch.float32)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        L1 = 256
        L2 = 128
        self.l1 = nn.Linear(state_dim + action_dim, L1)  # critic the value of Q(s,a)
        self.l2 = nn.Linear(L1, L2)
        self.l3 = nn.Linear(L2, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1))) # by cow
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)  # copy to gpu
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)  # to view

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #reshape(1,-1)变成只有一行的二维数组, 比如[1 2 3]变成[[1 2 3]]
        return self.actor(state).cpu().data.numpy().flatten()  # 转成nparray

    def update(self):
        for it in range(args.update_iteration):  # args.update_iteration=200
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device) # d=1 undone d=0 done
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value, it's off policy
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()
            # Get current Q estimate
            current_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')  # only save the prameters
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

#给外部文件调用agent
def use_agent(env, state):
    agent = DDPG(env.state_dim, env.action_dim, env.max_action)
    agent.load()
    action = agent.select_action(state)
    return action

def main():
    #运行主函数
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if args.mode == 'test':
        power_records_total = []
        temperature_records_total = []
        water_record_total = []
        agent.load()                
        for i in range(args.test_iteration):
            power_records = []
            temperature_records = []
            water_records = []

            actions = [[0]]
            state = env.reset()
            target_list = [1,6,10,21]
            env.today_signals = np.array([ [-1]*60 if time in target_list else [None]*60 for time in range(24)]).flatten()
            for t in count():
                action = agent.select_action(state)
                action = [0]
                actions.append(action)
                # print('action:')
                # print(action)
                next_state, reward, done, power_record, temperature_inside, water_mass = env.test_step(np.float32(actions[t]))
                power_records.append(power_record)
                temperature_records.append(temperature_inside)
                water_records.append(water_mass)
                
                ep_r += reward
                state = next_state
                if done:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
            power_records_total.append(np.array(power_records).flatten())
            temperature_records_total.append(np.array(temperature_records).flatten())
            # print(water_records)
            # print(len(water_records))
            # input()
            water_record_total.append(np.array(water_records).flatten())
            
        fig_, axes = plt.subplots(3, 5)
        for i in range(args.test_iteration):
            axes[0,i%5].plot(power_records_total[i])
            axes[1,i%5].plot(temperature_records_total[i])
            axes[2,i%5].plot(water_record_total[i])
        plt.show()
        
    elif args.mode == 'generate':
        
        agent.load()

        my_type = 0 if env.building.type=='office' else 1 if env.building.type=='commercial' else 2 if env.building.type=='hotel' else None
        my_area = env.building.area
        my_height = env.building.height
        my_layers = env.building.layers
        my_set_temperature = env.building.setted_temperature
        my_comfort = env.comfort
        my_attribute = np.array([my_type, my_area, my_height, my_layers, my_set_temperature, my_comfort])


        for month in [6,7,8,9,10]:
            power_records_total = []
            temperature_records_total = []

            if month in [6,9,11]:
                days = 30
            elif month in [3,7,8,10]:
                days = 31
            

            out_file_dir = "../data/" + args.building_type + '/' + str(args.random_seed) + '/'
            if not os.path.exists(out_file_dir):
                os.makedirs(out_file_dir)
            out_file_name = out_file_dir + str(month) + '.csv'
            

            for i in range(days):
                df = pd.DataFrame(columns=['type', 'area', 'height', 'layers', 'set_T', 'comfort_T', 'time', 'power', 'in_T', 'out_T', 'people_flow', 'base_power', 'capacity'])
                power_records = []
                temperature_records = []

                actions = [[0]]
                state = env.generate_reset(month, i)
                target_list = []
                env.today_signals = np.array([ [1]*60 if time in target_list else [None]*60 for time in range(24)]).flatten()
                for t in count():
                    action = agent.select_action(state)
                    actions.append(action)

                    my_state = state

                    my_data = np.append(my_attribute, np.append(my_state, action))
                    # print(df.shape)
                    # print(my_data.shape)
                    # print(my_data)
                    df.loc[df.shape[0]] = my_data
                    # print(my_attribute, type(my_attribute))
                    # print(my_state, type(my_state))
                    # print(action, type(action))
                    # input()

                    next_state, reward, done, power_record, temperature_inside = env.test_step(np.float32(actions[t]))
                    power_records.append(power_record)
                    temperature_records.append(temperature_inside)
                    # print(next_state)
                    # input()
                    ep_r += reward
                    state = next_state

                    if done:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                df.to_csv(out_file_name, mode='a', index=None, header=None)
                power_records_total.append(np.array(power_records).flatten())
                temperature_records_total.append(np.array(temperature_records).flatten())   

    elif args.mode == 'train':
        # agent.load()
        total_step = 0
        total_r = []
    
        for i in range(args.max_episode):  # max_episode = 100000
            total_reward = 0
            step =0
            state = env.reset()
    
            #state是一个一维的nparray
            for t in count():
                #根据状态选动作
                if i>30000:
                    args.exploration_noise = 0.1
                # add the noise in selecting action
                noise = np.random.normal(0, args.exploration_noise , size=env.action_dim)
                if t <= 5:
                    noise = 0
                action = (agent.select_action(state) + noise * max_action * 0.1).clip(0, max_action)
                print(noise * max_action * 0.1, agent.select_action(state))
    
                #环境根据动作给出下一个状态/当前时刻奖励/是否完成等信息
                next_state, reward, done = env.step(action)
    
                #存储样本
                agent.replay_buffer.store((state, next_state, action, reward, float(done)))
                #更新状态
                state = next_state
                step += 1
                total_reward += reward
    
                #input()
                
                # input()
                if done:
                    break
                
            #episode结束,记录reward
            total_step += step
            total_r.append(total_reward)
            print('----------------------------------')
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            if total_step > args.batch_size:
                agent.update()
    
            #一定时间后存储agent到文件中，方便下次调用或继续训练
            if i % args.save_interval == 0: #args.log_interval=50
                agent.save()
            
    else:
        raise NameError("mode wrong!!!")
    


# =============================================================================
if __name__ == '__main__':
    total_r = main()
# ============================================================================
