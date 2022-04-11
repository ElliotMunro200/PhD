# Defining the DQN architecture and update function along with other DQN helper functions.
###########################
import random
import numpy as np
from collections import deque
###########################
import torch
import torch.nn as nn
from torch.autograd import Variable
###########################
import wandb
###########################
# Replay buffer class, initially used for the Cartpole domain. It can be used generally though.
# Just need state, action, reward, next_state, done variables coming to it,
# and capacity variable for maximum size limit of the replay buffer deque.
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0) # tuple of shape (4,) -> ndarray of shape (1,4): e.g. array([[1,2,3,4]]).
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # zip(*...) performs an un-zip function so that a random batch is split into its original sample components.
        # N.B. each part has the length of the batch size.
        # May need to check these sizes and shapes if there are more problems beyond episodes not completing
        # when they should be.
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def epsilon_by_frame(frame_idx): # decaying exponential as function of frame index. Explore chance goes from 1.0->0.01.
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1. * frame_idx / epsilon_decay)
    return epsilon_by_frame

def train_log(metric_name, metric_val, frame_idx): # general function to log a metric with certain name to WandB
    # against its frame index.
    metric_val = float(metric_val)
    wandb.log({metric_name: metric_val}, step=frame_idx)
    print("Logging "+str(metric_name)+" of: "+str(metric_val)+", at frame index: "+str(frame_idx)+", to WandB")

class DQNet(nn.Module): #architecture of the DQN class is num_inputs=4 -> 256(ReLU) -> num_outputs=2.
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
        # state = list of state values e.g. looks like [1,2,3,4]
        # tensor output, exploiting old action value data.
        if random.random() > epsilon: # exploit
            state = torch.FloatTensor(state).unsqueeze(0) # torch equivalent of numpy.expand_dims(x,axis).
            # state now looks like tensor([[1,2,3,4]]).
            action = self.forward(state).max(1)[1]
            # mapping state to Q values by 'forward' and then doing max(1) to output (max,max_indicies) for axis=1,
            # then using [1] to grab the indicies tensor.
            return int(action[0]) # Grabbing the index in the tensor, and ensuring it is an int.
        # scalar output, exploring new actions for new action value data.
        else: # explore
            return random.randrange(self.num_outputs) # outputs random value 0 or 1.

def DQNet_update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) # each is the length of the batch size.
    # state = batch of states, action = batch of actions, etc.
    # state is 'numpy.ndarray', action is 'tuple', reward is 'tuple', next_state is 'numpy.ndarray', done is 'tuple'.

    # converting to torch tensors. torch.Tensor is an alias for the default tensor type torch.FloatTensor.
    # maybe can optimize in different situations by choosing specific tensor types.
    state = torch.Tensor(state)
    next_state = torch.Tensor(next_state)
    action = torch.LongTensor(action) # LongTensor because need ints for 'action.unsqueeze(1)'.
    reward = torch.Tensor(reward)
    done = torch.Tensor(done) # transforms booleans to False->0.0 and True->1.0. torch.Size([batch_size])

    q_value = model(state) # torch.Size([batch_size, num_outputs]). convert tensor to float by calling float(tensor).

    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1) # q_value: torch.Size([batch_size]).
    # action.unsqueeze(1): torch.Size([batch_size, 1]).
    # For gather: output[i][j][k] = input[i][index[i][j][k]][k], if dim == 1 (as here).
    # Here, i:1-batch_size, j=0, k=None (axis=2 doesn't exist).

    next_q_value = model(next_state).max(1)[0] # grabbing the actual max Q value, not the index of it as before.
    expected_q_value = reward + 0.99 * next_q_value * (1 - done) # done is batch of False->0.0 and True->1.0.
    loss = (q_value - expected_q_value.data).pow(2).mean() # MSE in Q values. single float tensor value with grad_fn.

    optimizer.zero_grad() # "Sets the gradients of all optimized 'torch.Tensor' s to zero."
    loss.backward() # "Computes the sum of gradients of given tensors with respect to [model] graph leaves."
    optimizer.step() # "Performs a single optimization step (parameter update)."
    return loss
    ############
