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
    print(state)
    for frame_idx in range(1, num_frames + 1):
        print("frame index:"+str(frame_idx))
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action) #print(next_state) --> StepType.MID
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
    env = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

    ###########################
    #Learning
    #1.Define learning network
    num_inputs = 4  # number of CartPole state variables
    num_outputs = 1  # number of CartPole action variables
    print(len(env.physics.data.qpos))
    print(len(env.physics.data.qvel))
    print(env.action_spec().shape[0])
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