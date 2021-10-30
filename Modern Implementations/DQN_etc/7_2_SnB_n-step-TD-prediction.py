import numpy as np
import random

class GridWorldBox:
    def __init__(self,grid_height,grid_length,start_pos,goal_pos,num_actions):
        self.grid_height = grid_height
        self.grid_length = grid_length
        self.goal_pos = goal_pos
        self.start_pos = start_pos
        self.num_actions = num_actions
        self.reset()

    def reset(self):
        self.end = False
        self.current_pos = [4,0]
        return self.current_pos

    def step(self,action):
        if action == 0:
            self.current_pos[0]-=1
        if action == 1:
            self.current_pos[1]-=1
        if action == 2:
            self.current_pos[1]+=1
        if action == 3:
            self.current_pos[0]+=1
        next_state = [self.current_pos[0],self.current_pos[1]]
        if self.current_pos[0]==self.goal_pos[0] and self.current_pos[1]==self.goal_pos[1]:
            self.end = True
            reward = 10
        else:
            reward = -1
        return next_state,reward,self.end

def policy(state):
    possible_actions = [a for a in range(Env.num_actions)] #start with all actions available
    if state[0] == 0:
        possible_actions.remove(0) #if in top row, can't go up (action=0)
    if state[0] == Env.grid_height-1:
        possible_actions.remove(3) #if in bottom row, can't go down (action=3)
    if state[1] == 0:
        possible_actions.remove(1) #if in leftmost column, can't go left (action=1)
    if state[1] == Env.grid_length-1:
        possible_actions.remove(2) #if in rightmost column, can't go right (action=2)
    action = random.choice(possible_actions) #select random action from possible actions
    return action

def nStep_TD_prediction():
    Vs = np.zeros((Env.grid_height,Env.grid_length))
    for ep in range(num_episodes):
        print("Episode number:"+str(ep))
        state = Env.reset()
        state_store = [state]
        reward_store = [0]
        T_ep = 1000000
        t=0
        tau=0
        while tau < T_ep:
            if t < T_ep:
                next_state,reward,done = Env.step(policy(state))
                state = next_state
                state_store.append(next_state)
                reward_store.append(reward)
                if done:
                    T_ep = t+1
            tau = t-n+1
            if tau >= 0:
                G=0
                for i in range(tau+1,min(tau+n,T_ep)+1):
                    G += ((gamma ** (i - tau - 1)) * reward_store[i])
                if tau+n<T_ep:
                    G += (gamma**n)*Vs[state_store[tau+n][0],state_store[tau+n][1]]
                Vs[state_store[tau][0], state_store[tau][1]] += alpha*(G-Vs[state_store[tau][0], state_store[tau][1]])
            t+=1
        #print("Number of episode timesteps:"+str(T_ep))
    return Vs

if __name__ == "__main__":
    #Env parameters and instantiation
    grid_height = 5
    grid_length = 5
    start_pos = [4,0]
    goal_pos = [0,4]
    num_actions = 4
    Env = GridWorldBox(grid_height,grid_length,start_pos,goal_pos,num_actions)
    #Learning parameters and learning
    alpha = 0.5
    gamma = 0.9
    n = 2
    num_episodes = 1000
    Vs = nStep_TD_prediction()
    print(Vs)