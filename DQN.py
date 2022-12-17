import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import rl_utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class ReplayBuffer:
    def __init__(self, bufferSize):
        self.buffer = collections.deque(maxlen=bufferSize)

    def add(self, state, action, reward, nextstate, done):
        self.buffer.append((state, action, reward, nextstate, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, nextstates, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(nextstates), dones

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, device, gamma, epsilon, learning_rate, update_num):
        self.qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        # self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.count = 0
        self.action_dim = action_dim
        self.update_num = update_num

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action

    def update(self, transitions):
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['nextstates'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)
        # next_q_values = self.qnet(next_states).max(1)[0].view(-1, 1)

        # next_q_values = self.target_qnet(next_states).max(1)[0].view(-1, 1)

        #DDQN
        a_star = self.qnet(next_states).max(1)[1].view(-1, 1)
        next_q_values = self.target_qnet(next_states).gather(1, a_star)

        loss = torch.mean(F.mse_loss(q_values, rewards + self.gamma * next_q_values * (1 - dones)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_num == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.count += 1




lr = 2e-3
batch_size = 64
buffer_size = 10000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
update_num = 10

num_episodes = 500
minimal_size = 500

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, device, gamma, epsilon, lr, update_num)
return_list = []

for i in range(10):
    with tqdm(total=num_episodes // 10, desc='Interation %d' % i) as par:
        for i_episodes in range(num_episodes // 10):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                # env.render()
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                episode_return += reward
                state = next_state

                if replay_buffer.size() > minimal_size:
                    bs, ba, br, bns, bd = replay_buffer.sample(batch_size)
                    transitions = {'states': bs,
                                  'actions': ba,
                                  'rewards': br,
                                  'nextstates': bns,
                                  'dones': bd}
                    agent.update(transitions)

            return_list.append(episode_return)

            if (i_episodes + 1) % 10 == 0:
                par.set_postfix({'Episode': '%d' % (i_episodes + 1),
                'Return': '%.3f' % np.mean(return_list[-10:])})

            par.update(1)


episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDQN on CartPloe-v0')
plt.show()



























