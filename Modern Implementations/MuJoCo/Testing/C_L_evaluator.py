#evaluate = Evaluator(args.validate_episodes,
#        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from C_L_util import *

class Evaluator(object): #typical use is during training to validate the training progress every (x=2000) timesteps.
    def __init__(self, num_episodes, interval, save_path="", max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)

    def __call__(self, env, policy, debug=False, visualize=False, save=False):
        self.is_training = False
        state = None
        result = []

        for episode in range(self.num_episodes): # num episodes per validate experiment for good stat. representation.

            # reset at the start of episode
            time_step = env.reset()
            state = time_step.observation
            positions = [x for x in state['unnamed_model/joint_positions'][0]]  # dim-positions = 8
            velocities = [x for x in state['unnamed_model/joint_velocities'][0]]  # dim-velocities = 8
            goal = env._task.task_observables['button_position']._raw_callable(env._physics)[:2]  # dim-rel_goal = 2
            state = [*positions, *velocities, *goal]  # extraction.
            episode_steps = 0
            episode_reward = 0.

            assert state is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(state)

                step_type, reward, discount, state = env.step(action)
                positions = [x for x in state['unnamed_model/joint_positions'][0]]  # dim-positions = 8
                velocities = [x for x in state['unnamed_model/joint_velocities'][0]]  # dim-velocities = 8
                goal = env._task.task_observables['button_position']._raw_callable(env._physics)[:2]  # dim-rel_goal = 2
                state = [*positions, *velocities, *goal]  # extraction.
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True

                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode, episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):
        y = np.mean(self.results, axis=0)
        error = np.std(self.results, axis=0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn + '.png')
        savemat(fn + '.mat', {'reward': self.results})