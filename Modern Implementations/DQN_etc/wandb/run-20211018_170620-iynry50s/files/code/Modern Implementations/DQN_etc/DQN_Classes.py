import random
from collections import deque
import numpy as np
from SDP_env_Class import SDP_env

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym
import wandb


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
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQNet, self).__init__()
        self.num_outputs = num_outputs
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        #state = onehot concatenation of state and goal
        #tensor output, exploiting
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(Variable(state)).max(1)[1]
            return int(action.data[0])
        #scalar output, exploring
        else:
            return random.randrange(self.num_outputs)


def to_onehot(x):
    oh = np.zeros(6)
    oh[x - 1] = 1.
    return oh


def epsilon_by_frame(frame_idx):
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)
    return epsilon_by_frame


def DQNet_update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(state))
    next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_value = model(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)

    next_q_value = model(next_state).max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_DQN(env,model,replay_buffer,batch_size,optimizer, num_frames, n_avg):
    losses = []
    all_rewards = []
    episode_reward = 0
    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        print("frame index:"+str(frame_idx))
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = DQNet_update(model, optimizer, replay_buffer, batch_size)
            losses.append(loss.data)

        #if frame_idx % num_frames == 0:
            #plot(frame_idx, all_rewards, losses)

    train_DQN_plot(all_rewards, n_avg, num_frames)

def train_DQN_plot(all_rewards, n, num_frames):
    plt.figure(figsize=(20, 5))
    plt.title("DQN: mean rewards over next "+str(n)+" episodes (" + str(num_frames) + " total frames)")
    plt.xlabel(str(n)+"th episode number")
    plt.ylabel("Rewards")
    # plot the mean reward over the next 100 episodes
    plt.plot([np.mean(all_rewards[i:i + n]) for i in range(0, len(all_rewards), n)])
    plt.show()


def train_h_DQN(env, meta_model, model, meta_replay_buffer, replay_buffer,
                batch_size,meta_optimizer, optimizer, num_frames, n_avg):
    state = env.reset()
    frame_idx = 1
    done = False
    all_rewards = []
    episode_reward = 0
    losses = []
    meta_losses = []

    while frame_idx < num_frames:
        print(frame_idx)
        goal = meta_model.act(state, epsilon_by_frame(frame_idx))
        onehot_goal = to_onehot(goal)

        meta_state = state
        extrinsic_reward = 0

        #training on frame index number until episode has ended (done=True)
        #or until the goal state is state 6.
        while not done and goal != np.argmax(state):
            goal_state = np.concatenate([state, onehot_goal])
            action = model.act(goal_state, epsilon_by_frame(frame_idx))
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            extrinsic_reward += reward
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

            replay_buffer.push(goal_state, action, intrinsic_reward, np.concatenate(
                [next_state, onehot_goal]), done)
            state = next_state

            model_loss = DQNet_update(model, optimizer, replay_buffer, batch_size)
            meta_model_loss = DQNet_update(meta_model, meta_optimizer,
                                           meta_replay_buffer, batch_size)
            losses.append(model_loss)
            meta_losses.append(meta_model_loss)
            frame_idx += 1

        meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)

        if done:
            state = env.reset()
            done = False
            all_rewards.append(episode_reward)
            episode_reward = 0

    train_h_DQN_plot(all_rewards, n_avg, num_frames)


def train_h_DQN_plot(all_rewards, n, num_frames):
    plt.figure(figsize=(20, 5))
    plt.title("h-DQN: mean rewards over next "+str(n)+" episodes (" + str(num_frames) + " total frames)")
    plt.xlabel(str(n)+"th episode number")
    plt.ylabel("Rewards")
    # plot the mean reward over the next 100 episodes
    plt.plot([np.mean(all_rewards[i:i + n]) for i in range(0, len(all_rewards), n)])
    plt.show()

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
    #possible_outputs = ["DQN_only","h-DQN_only","DQN_h-DQN_comparison"]
    DQN = True
    h_DQN = False
    num_frames = 5000
    batch_size = 32
    buffer_size = int(num_frames/10)
    n_avg = int(num_frames / 1000)

    config = {
        #"policy_type": "MlpPolicy",
        #"total_timesteps": 25000,
        "env_name": "CartPole-v0",
    }

    run = wandb.init(
        project="PyTorch", #name of project on WandB
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    #DQN
    if DQN:
        env = gym.make(config["env_name"]) #SDP_env()
        num_states = env.dim_states
        num_actions = env.dim_actions
        model = DQNet(num_states, num_actions)
        optimizer = optim.Adam(model.parameters())
        replay_buffer = ReplayBuffer(buffer_size)
        train_DQN(env,model,replay_buffer,batch_size,optimizer, num_frames, n_avg)

    #h-DQN
    if h_DQN:
        goal_state_rep_f = 2
        env = SDP_env()
        num_goals = env.dim_states
        num_actions = env.dim_actions
        model = DQNet(goal_state_rep_f*num_goals, num_actions)
        meta_model = DQNet(num_goals, num_goals)
        optimizer = optim.Adam(model.parameters())
        meta_optimizer = optim.Adam(meta_model.parameters())
        replay_buffer = ReplayBuffer(buffer_size)
        meta_replay_buffer = ReplayBuffer(buffer_size)
        train_h_DQN(env,meta_model,model,meta_replay_buffer,replay_buffer,
                    batch_size,meta_optimizer,optimizer,num_frames,n_avg)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#h-DQN control sequence

#imports
#instantiate environment-class object (init, reset, step(action))
#instantiate DQN-class model
#...(nn.Module parent class init, forward(x), act(state, epsilon))
#...and meta-model objects (normal and target for each)
#option to use GPU (add .cuda() to model objects
#create optimizer and meta-optimizer, replay buffer, and meta replay buffer
#define one-hot encoding function (index --> onehot vector: size=state-space)
#define update function(model, optimizer, buffer, batch_size)
#define epsilon(frame index) function using lambda
#define number frames(training length), initialize state, rewards
#while loop over frames:
#...meta-model.act gives a goal-action, then one-hot encode it
#...meta-state=one-hot encoded state, initialize extrinsic reward
#...while not done and meta-goal isn't reached:
#... ... make actions toward goal
#... ... update episode, extrinsic, and intrinsic rewards
#... ... push to replay buffer (goal-state, action, int-rew, new-goal-state, done)
#... ... update models with their optimizer and replay buffer
#... ... increment state and frame index
#... push to meta replay buffer(meta-state, goal, ext-rew, state, done)
#... if done: state=env.reset(), done=False, episode rewards append+reset
#plot episodic rewards (smoothed over next 100 episodes)