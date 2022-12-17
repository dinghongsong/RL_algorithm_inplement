import torch
import torch.nn as nn
import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
import torch.nn.functional as F
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Replay_buffer:
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

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        # print(cat.shape)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, device, tau, gamma, actor_lr, critic_lr, sigma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.sigma = sigma

    def take_action(self, state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transitions):
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)

        dones = torch.tensor(transitions['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_target = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


env_name = 'Pendulum-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
gamma = 0.98
tau = 0.005
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01

env.seed(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

replay_buffer = Replay_buffer(buffer_size)
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, device, tau, gamma, actor_lr, critic_lr, sigma)

return_list = []
for i in range(10):
    with tqdm(total=20, desc='Interation %d' % i) as pbar:
        for j in range(20):
            state = env.reset()
            episode_return = 0
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                episode_return += reward
                state = next_state

                if replay_buffer.size() > minimal_size:
                    bs, ba, br, bns, bd = replay_buffer.sample(batch_size)
                    transitions = {'states': bs, 'actions': ba, 'rewards':br, 'next_states':bns, 'dones':bd}
                    agent.update(transitions)

            return_list.append(episode_return)
            if (j + 1) % 10 == 0:
                pbar.set_postfix({'episode': (j + 1), 'return': np.mean(return_list[-10:])})
            pbar.update(1)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'DDPG ON {env_name}')
plt.show()














