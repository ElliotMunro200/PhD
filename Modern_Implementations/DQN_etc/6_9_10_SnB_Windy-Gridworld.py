import numpy as np

class WindyGridworld:
    def __init__(self):
        self.grid_height = 7
        self.grid_length = 10
        self.start_pos = [3,0]
        self.goal_pos = [3,7]
        self.num_actions = 4
        self.winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.current_pos = [3,0]
        self.reset()

    def reset(self):
        self.current_pos = [3,0]
        self.end = False
        return self.current_pos

    def step(self, action):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.current_pos[0] -= self.winds[self.current_pos[1]] #wind effect
        #action effects
        if action == 0:
            self.current_pos[0] -=1
        elif action == 1:
            self.current_pos[1] -=1
        elif action == 2:
            self.current_pos[1] +=1
        elif action == 3:
            self.current_pos[0] +=1
        elif action == 4:
            self.current_pos[0] -=1
            self.current_pos[1] -=1
        elif action == 5:
            self.current_pos[0] -=1
            self.current_pos[1] +=1
        elif action == 6:
            self.current_pos[0] +=1
            self.current_pos[1] -=1
        elif action == 7:
            self.current_pos[0] +=1
            self.current_pos[1] +=1
        elif action == 8:
            self.current_pos[0] +=0
            self.current_pos[1] +=0
        #checking doesn't go outside boundaries
        if self.current_pos[0] < 0:
            self.current_pos[0] = 0
        if self.current_pos[0] > self.grid_height-1:
            self.current_pos[0] = self.grid_height-1
        if self.current_pos[1] < 0:
            self.current_pos[1] = 0
        if self.current_pos[1] > self.grid_length-1:
            self.current_pos[1] = self.grid_length-1
        new_state = [self.current_pos[0],self.current_pos[1]]
        if new_state[0] == self.goal_pos[0] and new_state[1] == self.goal_pos[1]:#successfully got to goal state
            self.end = True
            reward = 0
        else:
            reward = -1
        return new_state, reward, self.end
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SARSA_Learning(num_episodes):
    Qs = np.zeros((WGEnv.grid_height,WGEnv.grid_length,WGEnv.num_actions))
    steps = []
    for ep in range(num_episodes):
        print(ep)
        state = WGEnv.reset()
        rand_num = np.random.random()
        if rand_num < epsilon: #explore
            action = np.random.randint(WGEnv.num_actions)
        else: #exploit
            action = np.argmax(Qs[state[0],state[1],:]) #find the greedy action for the given state
        step = 0
        done = WGEnv.end
        trajectory = [state]
        while not done: #continue learning until the episode ends
            state_dash, reward, done = WGEnv.step(action)
            trajectory.append(state_dash)
            step+=1
            rand_num2 = np.random.random()
            if rand_num2 < epsilon: # explore
                action_dash = np.random.randint(WGEnv.num_actions)
            else:  # exploit
                action_dash = np.argmax(Qs[state_dash[0], state_dash[1],:]) # find the greedy action for the given state
            Qs[state[0],state[1],action] = Qs[state[0],state[1],action] + alpha*(
                    reward+Qs[state_dash[0], state_dash[1],action_dash]-Qs[state[0],state[1],action])
            #print(state)
            state = WGEnv.current_pos
            action = action_dash
        steps.append(step)
    print(Qs)
    print(steps)
if __name__ == "__main__":
    #environment variables and env instantiation
    WGEnv = WindyGridworld()
    #epsilon-greedy SARSA variables and SARSA learning
    epsilon = 0.1
    alpha = 0.5
    num_episodes = 2000
    SARSA_Learning(num_episodes)



#Algorithm parameters: step size alpha = 0.5, epsilon = 0.1
#Initialize Q(s,a) = 0 for all s,a.

#Loop for each episode:
#... Initialize S
#... Choose A from S using e-greedy Q selection
#... Loop for each step of episode:
#... ... Take action A, observe R,S'
#... ... Choose A' from S' using e-greedy Q selection
#... ... Update Q(S,A) with SARSA update
#... ... S <- S'; A <- A';
#... until S is terminal