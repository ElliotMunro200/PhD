import numpy as np
from copy import deepcopy

import torch.nn as nn
from torch.optim import Adam

from S_L_CartPole_DDPG_model import (Actor, Critic)
from C_L_memory import SequentialMemory
from C_L_random_process import OrnsteinUhlenbeckProcess
from C_L_util import *

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, num_states, num_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.num_states = num_states
        self.num_actions = num_actions

        net_cfg = {
            'hidden1':args.hidden1,
            'hidden2':args.hidden2
        }

        self.actor = Actor(self.num_states, self.num_actions, **net_cfg)
        self.actor_target = Actor(self.num_states, self.num_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lrate)

        self.critic = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_target = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_optim = Adam(self.actor.parameters(), lr=args.lrate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions, theta=args.ou_theta, mu=args.ou_mu,
                                                       sigma=args.ou_sigma)

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.d_epsilon = 1.0/args.epsilon # 50,000
        self.discount = args.discount

        self.epsilon = 1.0
        self.s_t = None # most recent state
        self.a_t = None # most recent action
        self.is_training = True

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size) #using the memory data structures
        # Prepare for the target q batch
        t1 = torch.from_numpy(next_state_batch)
        t1 = t1.float()
        t2 = self.actor_target(deepcopy(t1))
        next_q_values = self.critic_target([t1,t2])

        target_q_batch = torch.Tensor(reward_batch) + \
                         self.discount * torch.Tensor(terminal_batch.astype(float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([torch.Tensor(state_batch), torch.Tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            torch.Tensor(state_batch),
            self.actor(torch.Tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update. default tau = 0.001
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return policy_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done) # note that s_t1 is not included in the memory
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.num_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(torch.Tensor(np.array([s_t])))
        ).squeeze(0)
        #print("action pre change: "+str(action))
        action += self.is_training * max(self.epsilon,0) * self.random_process.sample() * 5 # abs(x) ~< 0.1
        #print("action post explore+anneal: " + str(action))
        action = np.clip(action, -1., 1.)
        #print("action post clip: " + str(action))

        if decay_epsilon:
            self.epsilon -= self.d_epsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self, s):
        torch.manual_seed(s)