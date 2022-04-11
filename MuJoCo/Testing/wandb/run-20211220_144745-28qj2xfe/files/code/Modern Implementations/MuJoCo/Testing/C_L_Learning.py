#The main workflow of the CartPole example: instantiate env(task(environment+agent models))
#Composer
#1.Define agent
#2.Define environment
#3.Define task
#Learning
#1.Define learning network
#2.Define net update
#3.Define learning flow
###########################
import inspect

from dm_control import mujoco
from dm_control import mjcf
from dm_control import composer
from dm_control import suite
###########################
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import wandb
###########################
import torch.optim as optim
###########################
from C_L_CartPole_Agent import CartPole
from C_L_CartPole_Task_Env import CartPoleTask
from C_L_DQN import CartPole_ReplayBuffer
from C_L_DQN import CartPole_DQNet
from C_L_DQN import CartPole_DQNet_update
from C_L_DQN import epsilon_by_frame
from C_L_DQN import train_log
###########################
def train_CartPole_DQNet(env, model, replay_buffer, batch_size, optimizer, num_frames):
    wandb.watch(model, log="all", log_freq=10)
    losses = []
    all_rewards = []
    episode_reward = 0
    time_step = env.reset()
    state = time_step.observation
    state = ([state['position'][0], state['position'][2], state['velocity'][0], state['velocity'][1]])
    for frame_idx in range(1, num_frames + 1):
        print("frame index:"+str(frame_idx))
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)
        _, reward, done, obs = env.step(action)
        next_state = ([obs['position'][0], obs['position'][2],
                       obs['velocity'][0], obs['velocity'][1]])
        done = bool(time_step.discount)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            metric_name = "reward"
            train_log(metric_name, episode_reward, frame_idx)
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = CartPole_DQNet_update(model, optimizer, replay_buffer, batch_size)
            losses.append(loss.data)
            if frame_idx % (num_frames/100) == 0:
                metric_name = "loss"
                train_log(metric_name,loss,frame_idx)

if __name__ == "__main__":
    config = {
        "env_name": "MyCartPole",
        "buffer_size": 1000,
        "batch_size": 24,
        "max_frames_per_episode": 10000,
    }

    run = wandb.init(
        project="C_L"+config["env_name"],  #name of project on WandB
        config=config,
        job_type="learning",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )
    ###########################
    #Composer
    #1.Define agent
    #agent = CartPole()
    #2.Define environment & 3.Define task
    #task = CartPoleTask(agent)
    #env = composer.Environment(task, random_state=np.random.RandomState(42))
    random_state = np.random.RandomState(42)
    env = suite.load('cartpole', 'balance', task_kwargs={'random': random_state}) #suite.load is in suite.__init__.py
    #print(inspect.getmembers(env, inspect.ismethod)) #prints methods of env object in list of tuples

    ###########################
    #Learning
    #1.Define learning network
    num_inputs = 4  # number of CartPole state variables
    num_outputs = 1  # number of CartPole action variables
    #print(len(env.physics.data.qpos)) #2
    #print(len(env.physics.data.qvel)) #2
    #print(env.action_spec().shape[0]) #1
    net = CartPole_DQNet(num_outputs, num_outputs)
    #2.Define net update & 3.Define learning flow
    optimizer = optim.Adam(net.parameters())
    buffer_size = config["buffer_size"]
    replay_buffer = CartPole_ReplayBuffer(buffer_size)
    batch_size = config["batch_size"]
    num_frames = config["max_frames_per_episode"]
    train_CartPole_DQNet(env, net, replay_buffer, batch_size, optimizer, num_frames)
    ###########################
    run.finish()
    ###########################