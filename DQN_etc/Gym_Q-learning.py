from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import random

#https://gym.openai.com/envs/Taxi-v3/
#https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d

# the color blue indicates the passenger;
# magenta — the destination;
# yellow — our empty taxi;
# green — taxi when full;
# letters R, G, Y, B — locations.

#Fixing seed for reproducibility
np.random.seed(0)

#Loading, initializing, and rendering the gym environment
env = gym.make("Taxi-v3").env
env.reset()
env.render()

#Getting the state space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

#STEP 1: Initializing the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Setting the hyperparameters
alpha = 0.7  #learning rate
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
train_episodes = 2000
max_steps = 100

#Steps 2-5: Training the agent

#Creating lists to keep track of reward and epsilon values over the episodes
training_rewards_total = []
epsilons = []

#loop over all episodes for training
for episode in range(train_episodes):
    # Resetting the environment each time as per requirement
    state = env.reset()
    # Starting the tracker for the rewards
    training_rewards = 0

    #loop over all steps of a given episode
    for step in range(max_steps):

        ### STEP 2: SECOND option for choosing the initial action - exploit
        # If the random number is larger than epsilon: employing exploitation
        # and selecting best action available from the given state
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state, :])

            ### STEP 2: FIRST option for choosing the initial action - explore
        # Otherwise, employing exploration: choosing a random action
        else:
            action = env.action_space.sample()

        ### STEPs 3&4: performing the action, moving to new state, and getting the reward
        # Taking the action and getting the reward and outcome state
        new_state, reward, done, info = env.step(action)

        ### STEP 5: update the Q-table
        # Updating the Q-table using the Bellman equation
        Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        # Increasing our total reward and updating the state
        training_rewards += reward
        state = new_state

        # Ending the episode
        if done == True:
            # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
            break

    # Cutting down on exploration by reducing the epsilon after each episode
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    #Appending the total reward and reduced epsilon values after each episode
    training_rewards_total.append(training_rewards)
    epsilons.append(epsilon)

print("Average episode training score: " + str(sum(training_rewards_total) / train_episodes))

#Visualizing results and total reward over all episodes
x = range(train_episodes)
plt.plot(x, training_rewards_total)
plt.xlabel('Episode')
plt.ylabel('Training total reward')
plt.title('Total rewards over all episodes in training')
plt.show()

#Visualizing the epsilons over all episodes
plt.plot(epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title("Epsilon for episode")
plt.show()