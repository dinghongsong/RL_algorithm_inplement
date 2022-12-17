import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ValueNet1(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet1, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device, update_num):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.q_net = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet1(state_dim, hidden_dim).to(device)

        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
        self.gamma = gamma
        self.update_num = update_num
        self.count = 0

    def take_action(self, state):
        probs = self.policy_net(torch.tensor([state], dtype=torch.float).to(self.device))
        action_dict = torch.distributions.Categorical(probs)
        action = action_dict.sample()
        return action.item()

    def update(self, bs, ba, br, bns, bd):
        states = torch.tensor(bs, dtype=torch.float).to(self.device)
        actions = torch.tensor(ba).view(-1, 1).to(self.device)
        rewards = torch.tensor(br, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(bns, dtype=torch.float).to(self.device)
        dones = torch.tensor(bd,  dtype=torch.float).view(-1, 1).to(self.device)

        log_probs = torch.log(self.policy_net(states).gather(1, actions))
        # q_values = self.q_net(states).gather(1, actions)


        # a_star = self.q_net(next_states).max(1)[1].view(-1, 1)
        # next_q_value = self.target_q_net(next_states).gather(1, a_star)
        # td_target = rewards + self.gamma * next_q_value * (1 - dones)
        # td_loss = torch.mean(F.mse_loss(q_values, td_target.detach()))
        # td_delta = td_target - q_values
        td_target = rewards + self.critic(next_states) * self.gamma * (1 - dones)
        td_delta = td_target - self.critic(states)
        a = self.critic(states)
        critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.critic(states)))
        policy_loss = torch.mean(-log_probs * td_delta.detach())

        self.policy_net_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        policy_loss.backward()
        critic_loss.backward()
        self.policy_net_optimizer.step()
        self.critic_optimizer.step()


        # if self.count % self.update_num == 0:
        #     self.target_q_net.load_state_dict(self.q_net.state_dict())
        #
        # self.count += 1

env_name = 'CartPole-v0'
env = gym.make(env_name)

env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
actor_lr = 1e-3
critic_lr = 1e-2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gamma = 0.98
update_num = 10

agent = ActorCritic(state_dim, hidden_dim, action_dim,actor_lr, critic_lr, gamma, device, update_num)

return_list = []
for i in range(10):
    with tqdm(total=100, desc='Interation %d' % i) as par:
        for j in range(100):
            episode_return = 0
            state = env.reset()
            done = False
            bs, ba, br, bns, bd = [], [], [], [], []
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                bs.append(state)
                ba.append(action)
                br.append(reward)
                bns.append(next_state)
                bd.append(done)
                episode_return += reward
                state = next_state
            return_list.append(episode_return)
            agent.update(bs, ba, br, bns, bd)

            if (j + 1) % 10 == 0:
                par.set_postfix({'episode' : '%d' % (i * 100 + j + 1), 'return' : np.mean(return_list[-10:])})
            par.update(1)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'Actor-Critic on {env_name}')
plt.show()