import argparse

import gym
#from dm_control import suite
#from dm_control import mujoco
#from dm_control.mujoco.wrapper.mjbindings import enums

import spinup.algos.pytorch.ppo.ppo as ppo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch RL framework, Spinning Up PPO agent, ? env')
    parser.add_argument('--env', default='LunarLanderContinuous-v2', type=str, help='environment name') #HalfCheetah-v2
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--max_episode_time', default=10, type=int, help='max number of seconds in an episode')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='max number of steps in an episode')
    parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
    args = parser.parse_args()

    ppo = ppo(lambda : gym.make(args.env), seed=args.seed, max_ep_len=args.max_episode_length)
    #env_fn, actor_critic=<MagicMock spec='str' id='140554322637768'>, ac_kwargs={}, seed=0, steps_per_epoch=4000,
    # epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001, train_pi_iters=80, train_v_iters=80, lam=0.97,
    # max_ep_len=1000, target_kl=0.01, logger_kwargs={}, save_freq=10)

    # [optional] MuJoCo environment

    # env = suite.load('cartpole', 'balance', task_kwargs={
    #     'time_limit': args.max_episode_time,
    #     'random': args.seed})
    # printing environment timestep information
    # sim_timestep = env._physics.timestep()
    # print("simulation timestep: " + str(sim_timestep))
    # n_sub_steps = env._n_sub_steps
    # print("num sim sub-steps per control timestep: " + str(n_sub_steps))
    # ctrl_timestep = sim_timestep * n_sub_steps
    # print("control timestep: " + str(ctrl_timestep))
    # ep_time_limit = args.max_episode_time
    # print("episode time limit: " + str(ep_time_limit))
    # ep_step_limit = env._step_limit
    # print("episode step limit: " + str(ep_step_limit))
    # Visualize the joint axis
    # scene_option = mujoco.wrapper.core.MjvOption()
    # scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True