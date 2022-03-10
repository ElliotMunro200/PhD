###########################
import numpy as np
import argparse
from copy import deepcopy
###########################
from dm_control import suite
from dm_control import mujoco # The basic mujoco wrapper.
from dm_control.mujoco.wrapper.mjbindings import enums # Access to enums and MuJoCo library functions.
###########################
from S_L_CartPole_DDPG_evaluator import Evaluator
from C_L_ddpg import DDPG
from C_L_util import *
from DQN_Classes import train_log
from Matplotlib_Animation import display_video
###########################
import wandb
###########################
def train(num_iterations, agent, env, env_name, evaluate, validate_steps, output, framerate, use_wandb, make_animation,
          max_episode_length=None, debug=False):
    ######### FOR WANDB AND ANIMATIONS ###########
    if use_wandb:
        wandb.watch((agent.actor,agent.critic), log="all", log_freq=10) # "Hooks into the torch model (DQN) to collect gradients and the topology."
    frames = episode_frames = []
    losses = []
    all_rewards = []
    ##############################################
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    state = None
    while step < num_iterations: # total number of training iterations/steps
        if step % max_episode_length == 0:
            print("step: "+str(step))
        # reset if it is the start of episode
        if state is None:
            time_step = env.reset()
            state = deepcopy(time_step.observation)
            state = ([state['position'][0], state['position'][2], state['velocity'][0], state['velocity'][1]]) #extraction.
            agent.reset(state)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(state) # all necessary exploring behaviour is in here
        # interpret continuous action to discrete action-space CartPole environment
        if action < 0:
            action = -1
        else:
            action = 1

        # env response with next_observation, reward, terminate_info
        step_type, reward, discount, state2 = env.step(action)
        state2 = deepcopy(state2)
        state2 = ([state2['position'][0], state2['position'][2], state2['velocity'][0], state2['velocity'][1]])  # extraction.
        if reward == None:  # so that rewards can be summed.
            reward = 0.0
        print(state2[2])
        if abs(state2[2]) >= 0.5*np.sqrt(3): # if pole gets below 30 degrees from vertical on either side (x,v,t,w)
            env._reset_next_step = True
        done = env._reset_next_step
        print(done)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy, and WandB loss logging
        agent.observe(reward, state2, done)
        if step > args.warmup:
            policy_loss = agent.update_policy() # calculating the loss on a batch of data
            losses.append(policy_loss.data)  # tracking the loss data for each frame.
            if use_wandb and step % (num_iterations/max_episode_length) == 0:
                metric_name = "loss"
                train_log(metric_name, policy_loss, step)

        # [optional] evaluate.
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if use_wandb:
                metric_name = "validate_reward"
                train_log(metric_name, validate_reward, step)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] every 10th episode add all frames of the episode to the animation.
        if make_animation and episode % 10 == 0:
            if len(episode_frames) < env._physics.time() * framerate:
                # print("episode: "+str(episode))
                # print("len_frames: "+str(len(episode_frames)))
                # print(env._physics.time())
                # expected episode frames + previous episode frames
                # data.time resets each episode, so first frame should be recorded.
                pixels = env._physics.render(scene_option=scene_option, camera_id='lookatcart')
                # rendering the physics model scene into pixels. cameras defined in the cartpole.xml file in suite.
                episode_frames.append(pixels)  # building list of animation frames.
                # as soon as the 10th episode finishes, extend frames with the episode frames once, and reset.
                if done:
                    frames.extend(episode_frames)
                    episode_frames = []

        # [optional] save intermediate model.
        if step % int(num_iterations / 3) == 0:
            agent.save_model(output)

        # update at end of step.
        step += 1 # total steps
        episode_steps += 1 # episode steps
        episode_reward += reward # episode reward
        state = deepcopy(state2) # change default state

        if done:  # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
            all_rewards.append(episode_reward)  # tracking the total episode rewards across episodes.

            # WandB reward logging
            if use_wandb:
                metric_name = "reward"
                train_log(metric_name, episode_reward, step)  # logging the current total reward by frame index.

            agent.memory.append(
                state,
                agent.select_action(state),
                0., False
            )

            # reset
            state = None # calls for state reset
            episode_steps = 0 # reset episode steps counter
            episode_reward = 0. # reset episode reward counter
            episode += 1 # add to episode counter

    if make_animation:
        save_name = str(env_name) + "-" + str(num_iterations) + "_training_steps.mp4"
        print(save_name)
        display_video(save_name, frames)


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval() # puts agent DDPG layers in evaluation mode (changes behaviour of Dropout/BatchNorm layers if existing)
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='CartPole-v0', type=str, help='name of my dm_control composer env')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lrate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=20, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--batch_size', default=10, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=3, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=100, type=int, help='max number of steps in an episode')
    parser.add_argument('--max_episode_time', default=3, type=int, help='max number of seconds in an episode')
    parser.add_argument('--do_eval', default=False, type=bool, help='if to to perform a validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=2000, type=int, help='number of training iterations/timesteps')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--make_animation', default=True, type=bool, help='whether to make an animation for this run or not')
    parser.add_argument('--frate', default=30, type=int, help='framerate of animations')
    parser.add_argument('--wandb', default=False, type=bool, help='whether to invoke WandB for this run or not')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)
    ############# WANDB ###############
    # starts the WandB run. A WandB run is defined by the: project, config directory, job type, etc.
    if args.wandb:
        run = wandb.init(
            project="C_L_" + args.env,  # name of project on WandB
            config=args,
            job_type="DDPG_CartPole",
            save_code=True)  # optional
    ############# ENV #################

    env = suite.load('cartpole', 'balance', task_kwargs={
        'time_limit': args.max_episode_time,
        'random': args.seed})

    # Visualize the joint axis
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    #env = NormalizedEnv(gym.make(args.env)) <-- important. NormalizedEnv just wraps gym env for normalized actions.
    #env has a obs=env.reset(), obs,rew,done,info=env.step(action),
    #env.render(), env.seed(args.seed), dim_states=env.observation_space.shape[0], dim_actions=env.action_space.shape[0].

    ############# AGENT ###############

    if args.seed > 0:
        np.random.seed(args.seed)

    dim_states = 4
    dim_actions = 1

    agent = DDPG(dim_states, dim_actions, args)
    if args.do_eval:
        evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps, args.output, max_episode_length=args.max_episode_length)
    else:
        evaluate = None

    if args.mode == 'train':
        train(args.train_iter, agent, env, args.env, evaluate, args.validate_steps, args.output,
              args.frate, args.wandb, args.make_animation,
              max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
             visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

    if args.wandb:
        run.finish()
