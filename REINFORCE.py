import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
# import rl_utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PloicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PloicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, device, learning_rate):
        self.policy_net = PloicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        probs = self.policy_net(torch.tensor([state], dtype=torch.float).to(self.device))
        action_dict = torch.distributions.Categorical(probs=probs)
        action = action_dict.sample()
        return action.item()

    def update(self, transitions):
        states_list = transitions['states']
        actions_list = transitions['actions']
        rewards_list = transitions['rewards']

        u = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards_list))):
            u = self.gamma * u + rewards_list[i]
            state = torch.tensor([states_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor(actions_list[i]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            loss = -log_prob * u
            loss.backward()
        self.optimizer.step()


env_name = 'CartPole-v0'
env = gym.make(env_name)

env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
learning_rate = 1e-3
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent = REINFORCE(state_dim, hidden_dim, action_dim, gamma, device, learning_rate)

return_list = []
for i in range(10):
    with tqdm(total=100, desc='Iteration %d' % i) as par:
        for j in range(100):
            episode_return = 0
            state = env.reset()
            done = False
            transitions = {'states': [], 'actions': [], 'rewards': []}
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transitions['states'].append(state)
                transitions['actions'].append(action)
                transitions['rewards'].append(reward)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transitions) #每采集一条trajectory 就更新参数， 在线策略

            if (j + 1) % 10 == 0:
                par.set_postfix({'episode': '%d' % (i * 100 + j + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
            par.update(1)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'REINFORCE on {env_name}')
plt.show()
# learning_rate = 1e-3
# num_episodes = 1000
# hidden_dim = 128
# gamma = 0.98
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# env_name = "CartPole-v0"
# env = gym.make(env_name)
# env.seed(0)
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = REINFORCE(state_dim, hidden_dim, action_dim, gamma,
#                   device, learning_rate)
#
# return_list = []
# for i in range(10):
#     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             transition_dict = {
#                 'states': [],
#                 'actions': [],
#                 'next_states': [],
#                 'rewards': [],
#                 'dones': []
#             }
#             state = env.reset()
#             done = False
#             while not done:
#                 action = agent.take_action(state)
#                 next_state, reward, done, _ = env.step(action)
#                 transition_dict['states'].append(state)
#                 transition_dict['actions'].append(action)
#                 transition_dict['next_states'].append(next_state)
#                 transition_dict['rewards'].append(reward)
#                 transition_dict['dones'].append(done)
#                 state = next_state
#                 episode_return += reward
#             return_list.append(episode_return)
#             agent.update(transition_dict)
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode':
#                     '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)
#
# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('REINFORCE on {}'.format(env_name))
# plt.show()
#
# # mv_return = rl_utils.moving_average(return_list, 9)
# # plt.plot(episodes_list, mv_return)
# # plt.xlabel('Episodes')
# # plt.ylabel('Returns')
# # plt.title('REINFORCE on {}'.format(env_name))
# # plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
