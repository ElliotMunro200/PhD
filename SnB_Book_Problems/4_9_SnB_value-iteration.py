# Exercise 4.9 (programming) Implement value iteration for the gambler’s problem and
# solve it for p h = 0.25 and p h = 0.55. In programming, you may find it convenient to
# introduce two dummy states corresponding to termination with capital of 0 and 100,
# giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.
# Are your results stable as ✓ ! 0?

from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

#initializes V and pi
def init():
    V_s = np.random.rand(101)
    V_s[0]=0
    V_s[100]=1
    return V_s

#The p-function (normal function of 4 variables) - outputs a probability of s' and r given s and a
def p_function(s,a,V_s,p_h=0.4):
    s_dash_lose = s - a
    r = 0
    Q_lose = (1-p_h)*(r+V_s[s_dash_lose])
    s_dash_win = s+2*a
    if s_dash_win == 100:
        r=1
    else:
        r=0
    Q_win = p_h*(r+V_s[s_dash_win])
    Q = Q_win+Q_lose
    print(Q)
    return Q

#performs value iteration
def train(V_s, theta, p_h):
    delta = theta + 1
    while delta > theta:
        for i,s in enumerate(V_s):
            if i == 0 or i == 100:
                break
            v = V_s[i]
            V_s_is = []
            s = int(s)
            for a in range(min(s,100-s)+1):
                V_s_i = p_function(s, a, V_s, p_h)
                V_s_is.append(V_s_i)
            V_s[i] = max(V_s_is)
            delta = max(delta, abs(v - V_s[i]))
    pi_s = np.zeros(100)
    for i,s in enumerate(V_s[1:100]):
        Q = lambda a: p_function(s, a, V_s, p_h)
        pi_s[i] = max(range(min(s,100-s)+1), key=Q)
    return pi_s, V_s

def draw(pi_s):
    plt.plot(range(100), pi_s)
    plt.show()

def main():
    theta = 0.0001
    p_h_vals = [0.4,0.25,0.55]
    for p_h in p_h_vals:
        V_s = init()
        pi_s = train(V_s, theta, p_h)
        draw(pi_s)

if __name__ == "__main__":
    main()