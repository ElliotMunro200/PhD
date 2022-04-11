from collections import deque
from PhD.SnB_Book_Problems.SDP_env_Class import SDP_env

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        if len(state[0])!=1:
            state = np.concatenate(state)
            next_state = np.concatenate(next_state)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQNet, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(num_inputs, 128),
        nn.ReLU(),
        nn.Linear(128, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)

    #state is normal scalar value
    def act(self, state, epsilon):
        #explore
        if random.random() < epsilon:
            action_list = random.sample(env1.action_space(), 1)
            action = action_list[0]
        #exploit
        else:
            if state == 6:
                action = 0
                return action
            state = Variable(torch.FloatTensor(np.float32([state])))
            q_value = self.forward(state)
            #print(q_value)
            action = q_value.detach().numpy().argmax()
            #print(action)
        return action

def DQNet_update(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(np.float32([state])))
    next_state = Variable(torch.FloatTensor(np.float32([next_state])))
    action = Variable(torch.LongTensor([action]))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values      = model(state).squeeze(0)
    next_q_values = model(next_state).squeeze(0)
    #print("q-values:"+str(q_values))
    #print("action:"+str(action))
    #print("action.unsqueeze:"+str(action.unsqueeze(1)))
    q_value = q_values.gather(1, action).squeeze(1)
    next_q_values = next_q_values.squeeze(0)
    next_q_value = next_q_values.max(1)[0]
    #print(reward.shape, next_q_value.shape, done.shape)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title("frame %s. reward: %s" % (frame_idx, np.mean(rewards[-1000:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title("loss")
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    #instantiate replay buffer, SDP environment, and DQN model, and choose optimizer
    buffer_size = 1000
    replay_buffer = ReplayBuffer(buffer_size)
    state_list = [1, 2, 3, 4, 5, 6]
    env1 = SDP_env(state_list)
    state = env1.state
    model = DQNet(len([state]), 2)
    optimizer = optim.Adam(model.parameters())
    #define decaying epsilon value as a function of the frame index
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)
    #define parameters and train the DQN model
    num_frames = 10000
    batch_size = 10
    gamma = 0.99
    losses = []
    all_rewards = []
    episode_reward = 0
    for frame_idx in range(1, num_frames + 1):
        print("frame index:"+str(frame_idx))
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done = env1.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        #print("replay buffer length:"+str(replay_buffer.__len__()))


        state = next_state
        episode_reward += reward

        if done:
            state = env1.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = DQNet_update(batch_size)
            losses.append(loss.data)

        if frame_idx % num_frames == 0:
            plot(frame_idx, all_rewards, losses)
