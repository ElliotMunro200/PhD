import numpy as np
import random
import matplotlib.pyplot as plt
from SDP_env_Class import SDP_env

def from_onehot_to_scalar(onehot_vec):
    scalar = np.argmax(onehot_vec)+1
    return scalar

def epsilon_decay(episode):
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.001
    epsilon = min_epsilon + (max_epsilon-min_epsilon) * np.exp(-decay * episode)
    return epsilon

def Q_learning_run(env, num_episodes, steps_per_episode):
    #initialize parameters
    alpha = 0.25
    discount = 0.99
    #initialize Q-values, rewards, epsilons for training
    Qs = np.zeros((env.num_states, env.num_actions))
    rewards_by_episode = []
    epsilons = []
    for episode in range(num_episodes):
        #initialize episode
        env.reset()
        state = env.current_state
        episode_rewards = 0
        epsilon = epsilon_decay(episode)
        #perform the steps for a given episode
        for step in range(steps_per_episode):
            #pick the action for a given step with epsilon-greedy action selection
            rand_num = random.random()
            #explore
            if rand_num < epsilon:
                action = random.sample(range(env.num_actions),1)
            #exploit
            else:
                action = np.argmax(Qs[state-1,:])
            #take step according to chosen action
            new_state, reward, done, _ = env.step(action)
            new_state = from_onehot_to_scalar(new_state)
            #update Q-table
            Qs[state-1, action] = Qs[state-1, action] + alpha * (
                    reward + discount * np.max(Qs[new_state-1, :]) - Qs[state-1, action])
            #reward counting and state shift
            episode_rewards += reward
            state = new_state
            #end episode or not?
            if step == steps_per_episode - 1:
                done = True
            if done == True:
                rewards_by_episode.append(episode_rewards)
                break

        epsilons.append(epsilon)

        if episode%100 == 0:
            print("Total reward for episode {}: {}".format(episode, episode_rewards))
    print(Qs)
    return rewards_by_episode, epsilons

def Q_learning_run_plot(num_episodes, rewards_by_episode, epsilons):
    episodes = range(num_episodes)
    plt.plot(episodes, rewards_by_episode)
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("")
    plt.title("Q-learning performance")
    plt.legend(["Episode Reward", "Epsilons"])
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def DQN_learning_run():
    pass


def DQN_learning_run_plot(num_episodes, rewards_by_episode, epsilons):
    episodes = range(num_episodes)
    plt.plot(episodes, rewards_by_episode)
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("")
    plt.title("DQN-learning performance")
    plt.legend(["Episode Reward", "Epsilons"])
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    #define the Q-learning run
    num_episodes = 10000
    steps_per_episode = 50
    env1 = SDP_env()
    #Run
    rewards_by_episode, epsilons = Q_learning_run(env1, num_episodes, steps_per_episode)
    Q_learning_run_plot(num_episodes, rewards_by_episode, epsilons)