from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

#Write a program for policy iteration and re-solve Jack’s car
# rental problem with the following changes. One of Jack’s employees at the first location
# rides a bus home each night and lives near the second location. She is happy to shuttle
# one car to the second location for free. Each additional car still costs $2, as do all cars
# moved in the other direction. In addition, Jack has limited parking space at each location.
# If more than 10 cars are kept overnight at a location (after any moving of cars), then an
# additional cost of $4 must be incurred to use a second parking lot (independent of how
# many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
# occur in real problems and cannot easily be handled by optimization methods other than
# dynamic programming. To check your program, first replicate the results given for the
# original problem.

import numpy as numpy

#initializes V and pi
def init(n_c_p_l):
    V_s = np.random.rand(n_c_p_l, n_c_p_l)
    pi_s = np.random.zeros(n_c_p_l, n_c_p_l)
    return V_s, pi_s

#The p-function (normal function of 4 variables) - outputs a probability of s' and r given s and a
def p_function(i,a,V_s,n_c_p_l):
    num_1 = i//n_c_p_l
    num_2 = (i%n_c_p_l)+1
    state = (num_1, num_2)
    action = a
    for n in range(11): #n is the number of cars requested (lambda = 3, 4) or returned (lambda = 3, 2)
        poisson.pmf(n, lambda_cars) #calculates the requests and returns of cars (Poisson random variables)



    return V

#performs policy evaluation - finds the value function of the current policy using it and the old value function
def policy_evaluation(V_s, pi_s, n_c_p_l):
    theta = 5
    delta = theta + 1
    V_s = V_s.flatten("C")
    pi_s = pi_s.flatten("C")
    while delta >= theta:
        delta = 0
        for i,s in enumerate(V_s):
            v = V_s[i]
            V_s[i] = p_function(i,pi_s[i],V_s,n_c_p_l)
            delta = max(delta, abs(v-V_s[i]))
    V_s = V_s.reshape((n_c_p_l, n_c_p_l), "C")
    return V_s

#performs policy improvement - makes a new greedy policy w.r.t the value function of the current policy
def policy_improvement(pi_s, V_s, n_c_p_l):
    policy_stable = True
    pi_s = pi_s.flatten("C")
    V_s = V_s.flatten("C")
    for i,s in enumerate(V_s):
        old_action = pi_s
        Q = lambda a: p_function(i,a,V_s,n_c_p_l) #mapping a to Q(s,a) through the p-function
        pi_s[i] = max(range(11), key=Q) #finding best a (deterministic policy) by argmax_a of Q(s,a) using mapping
        if old_action[i] != pi_s[i]:
            policy_stable = False
    if policy_stable:
        pi_s = pi_s.reshape((n_c_p_l, n_c_p_l), "C")
        return pi_s, policy_stable

#performs policy iteration
def train(V_s, pi_s, n_c_p_l):
    policy_stable = False
    while not policy_stable:
        V_s = policy_evaluation(V_s, pi_s, n_c_p_l)
        pi_s, policy_stable = policy_improvement(pi_s, V_s, n_c_p_l)
    return V_s, pi_s

def main(n_c_p_l):
        V_s, pi_s = init(n_c_p_l)
        V_s_star, pi_s_star = train(V_s, pi_s, n_c_p_l)
        draw(V_s_star, pi_s_star)

    if __name__ == "__main__":
        main(n_c_p_l=20)