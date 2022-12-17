import gym
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import collections
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)

class VAnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)
        self.fc_A = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        V = self.fc_V(F.relu(self.fc1(x)))
        A = self.fc_A(F.relu(self.fc1(x)))
        dd = A.mean(1)
        ddd = A.mean(1).view(-1, 1)
        Q = V + A - A.mean(1).view(-1, 1)  ##############
        return Q

class DuelingDQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device, epsilon, update_num):
        self.qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        self.epsilon = epsilon
        self.update_num = update_num
        self.action_dim = action_dim
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.qnet(torch.tensor([state], dtype=torch.float).to(self.device)).argmax(1).item()
        return action

    def max_q_value(self, state):
        return self.qnet(torch.tensor([state], dtype=torch.float).to(self.device)).max(1).values.item()

    def update(self, bs, ba, br, bns, bd):
        states = torch.tensor(bs, dtype=torch.float).to(self.device)
        actions = torch.tensor(ba).view(-1, 1).to(self.device)
        rewards = torch.tensor(br, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(bns, dtype=torch.float).to(self.device)
        dones = torch.tensor(bd, dtype=torch.float).view(-1, 1).to(self.device)

        #DDQN
        q_values = self.qnet(states).gather(1, actions)
        a_star = self.qnet(next_states).max(1)[1].view(-1, 1)
        next_q_value = self.target_qnet(next_states).gather(1, a_star)

        TD_loss = torch.mean(F.mse_loss(q_values, rewards + self.gamma * next_q_value * (1 - dones)))

        self.optimizer.zero_grad()
        TD_loss.backward()
        self.optimizer.step()

        if self.count % self.update_num == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.count += 1


env_name = 'CartPole-v0'
env = gym.make(env_name)

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
learning_rate = 2e-3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gamma = 0.98
epsilon = 0.01
update_num = 10
batch_size = 64

replay_buffer = ReplayBuffer(buffer_size=10000)
agent = DuelingDQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, device, epsilon, update_num)

return_list = []
for i in range(10):
    with tqdm(total=50, desc='Interation %d' % i) as par:
        for j in range(50):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_return += reward


                if replay_buffer.size() > 500:
                    bs, ba, br, bns, bd = replay_buffer.sample(batch_size)
                    agent.update(bs, ba, br, bns, bd)

            return_list.append(episode_return)
            if j % 10 == 0:
                par.set_postfix({'episode': '% d' % j, 'return' : '%d' % np.mean(return_list[-10:])})
            par.update(1)

episode = list(range(len(return_list)))
plt.plot(episode, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'Dueling DQN on {env_name}')
plt.show()



































