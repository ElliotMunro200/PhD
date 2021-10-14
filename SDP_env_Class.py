import random
import numpy as np

class SDP_env:
    def __init__(self):
        self.num_actions = 2
        self.num_states = 6
        self.p_right = 0.5
        self.reset()

    def reset(self):
        self.end = False
        self.current_state = 2
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action):
        if self.current_state != 1:
            if action == 1:
                if random.random() < self.p_right and self.current_state < self.num_states:
                    self.current_state += 1
                else:
                    self.current_state -= 1

            if action == 0:
                self.current_state -= 1

            if self.current_state == self.num_states:
                self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.

        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00 / 100.00, True, {}
        else:
            return state, 0.0, False, {}

if __name__ == "__main__":
    number_s6 = 0
    for i in range(100000):
        env1 = SDP_env()
        while env1.current_state > 1:
            action = random.sample(range(env1.num_actions),1)
            s,r,d,_ = env1.step(action[0])
        if r == 1.0:
            number_s6 +=1
    print(number_s6)



