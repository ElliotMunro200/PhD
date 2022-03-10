# The main workflow of the CartPole example: instantiate env(task(environment+agent models))
# Composer
# 1.Define agent
# 2.Define environment
# 3.Define task
# Learning
# 1.Define learning network
# 2.Define net update
# 3.Define learning flow
###########################
import numpy as np
import statistics as stats
import argparse
from copy import deepcopy
###########################
from dm_control import suite
from dm_control import mujoco
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
import wandb
###########################
import torch.optim as optim
###########################
from DQN_Classes import ReplayBuffer
from DQN_Classes import DQNet
from DQN_Classes import DQNet_update
from DQN_Classes import epsilon_by_frame
from DQN_Classes import train_log
from Matplotlib_Animation import display_video


###########################
def train_CartPole_DQNet(env, env_name, model, replay_buffer, batch_size, optimizer, num_iterations,
                         use_wandb, make_animation, framerate):
    # initialization
    if use_wandb:
        # "Hooks into the torch model (DQN) to collect gradients and the topology."
        wandb.watch(model, log="all", log_freq=10)
    frames = []
    episode_frames = []
    num_frames = []
    losses = []
    all_rewards = []
    episode_reward = 0
    episode = 0
    state = None

    # Defining the project run for loop of frames. Each frame there is: action selection; state transition; observation;
    # end of episode check; replay buffer push; logging of: rewards (if end of episode), loss (if multiple of a 100th
    # of the way through the project run); DQN update if there are enough samples in the buffer.
    for step in range(1, num_iterations + 1):
        if step % (num_iterations/100) == 0:
            print("frame index:" + str(step))

        # resetting state in case of new episode.
        if state is None:
            # env.reset() starts a new episode and returns 'TimeStep'(step-type, reward, discount, observation).
            time_step = env.reset()
            state = time_step.observation
            state = ([state['position'][0], state['position'][2], state['velocity'][0], state['velocity'][1]])  # extraction.

        # action selection
        epsilon = epsilon_by_frame(step)  # value of decaying exponential as function of frame index.
        action = model.act(state, epsilon)  # action selected as function of explore probability epsilon and the state.
        # convert actions from 0/1 to -1/1 for use with the environment. They are used as 0/1 later as indicies.
        action_env = deepcopy(action)
        if action_env == 0:
            action_env = -1

        # action taken and new state, rewards observed.
        _, reward, discount, obs = env.step(action_env)  # action taken. returns step-type, reward, discount, obs.
        next_state = ([obs['position'][0], obs['position'][2], obs['velocity'][0], obs['velocity'][1]])
        if reward == None:  # so that rewards can be summed.
            reward = 0.0

        # determining if this new state requires an episode reset.
        done = env._reset_next_step  # Boolean variable of whether episode is finished or not.
        if abs(next_state[1]) >= 0.5 * np.sqrt(3):  # if pole gets below 30 degrees from vertical on either side
            done = True

        # pushing this most recent transition to the replay buffer.
        replay_buffer.push(state, action, reward, next_state, done)  # pushing the sample to the replay buffer.

        #print("action: " + str(action))
        #print("pole-angle: " + str(next_state[2]))
        #print("step-type: " + str(_))  # StepType.MID

        # updating the current state and the episode rewards count
        state = next_state
        episode_reward += reward

        # [optional] animation creation
        if make_animation and episode % 10 == 0:
            if len(episode_frames) < env._physics.data.time * framerate:
                pixels = env._physics.render(scene_option=scene_option, camera_id='lookatcart')
                # rendering the physics model scene into pixels. cameras defined in the cartpole.xml file in suite.
                episode_frames.append(pixels)  # building list of animation frames.
            if done: # as soon as the 10th episode finishes, extend frames with the episode frames once, and reset.
                frames.extend(episode_frames)
                num_frames.append(len(episode_frames))
                episode_frames = []

        # after-step model update and [optional] loss logging
        if len(replay_buffer) > batch_size:
            loss = DQNet_update(model, optimizer, replay_buffer, batch_size)  # calculating the loss on a batch
            # of data from the replay buffer according to the DQN update function in 'DQN_Classes.py'.
            losses.append(loss.data)
            # [optional] WandB loss logging
            if use_wandb and step % (num_iterations / 100) == 0:
                metric_name = "loss"
                train_log(metric_name, loss, step)

        # end of episode resetting and [optional] reward logging
        if done:
            all_rewards.append(episode_reward)  # tracking the total episode rewards across episodes.
            # [optional] WandB reward logging
            if use_wandb:
                metric_name = "reward"
                train_log(metric_name, episode_reward, step)  # logging the current total reward by frame index.
            episode_reward = 0
            episode += 1
            state = None

    if make_animation:
        save_name = str(env_name) + "-" + str(num_iterations) + "_training_steps.mp4"
        print("Total frame length of animation: " + str(len(frames)))
        print("Episodes of animation: " + str(((episode - 1) // 10) + 1))
        print("Number of frames by animation episode (10th): " + str(num_frames))
        print("Mean of all episode rewards: " + str(stats.mean(all_rewards)))
        print(save_name)
        display_video(save_name, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--use_wandb', default=False, type=bool, help='whether to invoke WandB for this run or not')
    parser.add_argument('--make_animation', default=True, type=bool, help='whether to make an animation this run or not')
    parser.add_argument('--env_name', default="MyCartPole", type=str, help='environment name for naming project')
    parser.add_argument('--buffer_size', default=100, type=int, help='buffer size')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--num_iterations', default=1000, type=int, help='number of total frames in the run')
    parser.add_argument('--ep_time_limit', default=10, type=int, help='episode time limit in seconds')
    parser.add_argument('--framerate', default=30, type=int, help='framerate for animation frame capture')
    args = parser.parse_args()

    # starts the WandB run. A WandB run is defined by the: project, config directory, job type, etc.
    if args.use_wandb:
        run = wandb.init(
            project="S_L" + args.env_name,
            config=args,
            job_type="learning",
            save_code=True,  # optional
        )

    random_state = np.random.RandomState(2)
    # loads a pre-defined environment constructed from the domain(agent+environment physics models)/task combination
    # from the dm_control suite library.
    env = suite.load('cartpole', 'balance', task_kwargs={
        'time_limit': args.ep_time_limit,
        'random': random_state})
    # Visualize the joint axis
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    # environment timestep information
    sim_timestep = env._physics.timestep()
    print("simulation timestep: "+str(sim_timestep))
    n_sub_steps = env._n_sub_steps
    print("num sim sub-steps per control timestep: "+str(n_sub_steps))
    ctrl_timestep = sim_timestep*n_sub_steps
    print("control timestep: "+str(ctrl_timestep))
    ep_time_limit = args.ep_time_limit
    print("episode time limit: "+str(ep_time_limit))
    ep_step_limit = env._step_limit
    print("episode step limit: "+str(ep_step_limit))

    # Learning
    # 1.Define learning network
    num_inputs = 4  # the number of CartPole state variables: (x, v, theta, omega) = (cart+pole, pos+vel).
    num_outputs = 2  # the number of the CartPole action variable options: left and right. A = argmax_a_Q(s,a)
    # print(len(env.physics.data.qpos)) # 2 position variables.
    # print(len(env.physics.data.qvel)) # 2 velocity variables.
    # print(env.action_spec().shape[0]) # 1 action variable (discrete valued: left [-1] and right [1]).
    # defining the agent Q network architecture. Inputs the full state, outputs the action-values Q(s,a).
    net = DQNet(num_inputs, num_outputs)

    # 2.Define net update & 3.Define learning flow
    optimizer = optim.Adam(net.parameters())  # Optimizer Adam uses DQN loss gradients in its way for backpropagation.
    replay_buffer = ReplayBuffer(args.buffer_size)  # instantiating the replay buffer.
    train_CartPole_DQNet(env, args.env_name, net, replay_buffer, args.batch_size, optimizer, args.num_iterations,
                         args.use_wandb, args.make_animation, args.framerate)
    if args.use_wandb:
        run.finish()
