import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, gamma, lmbda, epochs, eps):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, tranditions):
        states = torch.tensor(tranditions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(tranditions['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(tranditions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(tranditions['next_states'], dtype=torch.float).to(self.device)

        dones = torch.tensor(tranditions['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = torch.tensor(td_delta, dtype=torch.float).to(self.device)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_probs = self.actor(states).gather(1,actions).detach()
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            probs = self.actor(states).gather(1, actions)
            ratio = probs / old_probs
            # ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            suur2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # actor_loss = torch.mean(- torch.min(suur2, surr1))
            actor_loss = torch.mean(-suur2)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env.seed(0)
torch.manual_seed(0)
agent = PPO(state_dim, hidden_dim, action_dim,  actor_lr, critic_lr, device, gamma, lmbda, epochs, eps)

return_list = []
for i in range(10):
    with tqdm(total=50, desc='Interation %d' % i) as pbar:
        for j in range(50):
            episode_return = 0
            state = env.reset()
            done = False
            transitions = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transitions['states'].append(state)
                transitions['actions'].append(action)
                transitions['rewards'].append(reward)
                transitions['next_states'].append(next_state)
                transitions['dones'].append(done)

                episode_return += reward
                state = next_state
            agent.update(transitions)
            return_list.append(episode_return)
            if (j + 1) % 10 == 0:
                pbar.set_postfix({'episode': j + 1, 'return': np.mean(return_list[-10:])})
            pbar.update(1)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'PPO on {env_name}')
plt.show()
















