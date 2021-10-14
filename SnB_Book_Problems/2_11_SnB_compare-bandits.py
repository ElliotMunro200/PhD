import random
import numpy as np
from matplotlib import pyplot as plt
import timeit

#sample average and constant update step for nonstationary k-arm bandit problem.
# constant step allows for better tracking of nonstationary q* values because it doesn't converge.
# Make a figure analogous to Figure 2.6. Use runs of 200,000 steps and, as a performance measure for each algorithm and
# parameter setting, use the average reward over the last 100,000 steps.

k_arms = 10
n_steps = 200000
runs = 10
qstars_mu = 0
qstars_sd = 1
qstars_mu_add = 0
qstars_sd_add = 0.01
alpha = 0.1
params = []
R_n_all = []
R_n_all_c = []

def bandit(A):
    R = np.random.normal(qstars[A],1)
    return R

runs = range(runs)
for r in runs:
    print(r)
    epsilon = (1/128)*2**r
    #one run for step size = 1/n
    qstars = np.zeros(k_arms)
    for qi in range(k_arms):
        qs = np.random.normal(qstars_mu, qstars_sd)
        qstars[qi] = qs
    #all estimates of q* will be entered for each step, even if they haven't changed
    Q_a = np.zeros((n_steps,k_arms))
    Q_a_c = np.zeros((n_steps, k_arms))
    R_n = np.zeros(n_steps)
    R_n_c = np.zeros(n_steps)
    N_a = [0] * k_arms
    N_a_c = [0] * k_arms

    i_step = 1
    #n_step search
    while i_step <= n_steps:
        #finding the action that is currently estimated as the best value
        Q = lambda a: Q_a[i_step - 1][a]
        A1 = max(range(len(Q_a[i_step - 1])), key=Q)
        Q_c = lambda a_c: Q_a_c[i_step - 1][a_c]
        A1_c = max(range(len(Q_a_c[i_step - 1])), key=Q_c)
        #exploiting the best action
        if random.random() > epsilon:
            A = A1
            A_c = A1_c
        #exploring with a random action
        else:
            A = random.randint(0, k_arms - 1)
            A_c = random.randint(0, k_arms - 1)
        #generate the reward from the bandit
        R = bandit(A)
        R_c = bandit(A_c)
        #increment the number of times that bandit is chosen
        N_a[A] += 1
        N_a_c[A_c] += 1
        #make q* values drift over time
        for i,qstar in enumerate(qstars):
            qstars[i] = qstar + np.random.normal(qstars_mu_add, qstars_sd_add)
        #updating q* estimate (Q) values
        if i_step < n_steps:
            #current step = last step
            Q_a[i_step][:] = Q_a[i_step-1][:]
            Q_a_c[i_step][:] = Q_a_c[i_step - 1][:]
            #calculate new value update with incremental sample average
            Q_A_new = Q_a[i_step-1][A] + 1/float(N_a[A])*(R-Q_a[i_step-1][A])
            Q_A_new_c = Q_a_c[i_step-1][A_c] + alpha*(R_c-Q_a_c[i_step - 1][A_c])
            #insert new value estimate
            Q_a[i_step][A] = Q_A_new
            Q_a_c[i_step][A_c] = Q_A_new_c
            # add new average reward (as incremented sample average) for this episode
            if i_step > n_steps/2:
                R_n[i_step] = R_n[i_step - 1] + 1 / float(i_step-n_steps/2) * (R - R_n[i_step - 1])
                R_n_c[i_step] = R_n_c[i_step - 1] + 1 / float(i_step-n_steps/2) * (R_c - R_n_c[i_step - 1])
        #increment the step counter
        i_step += 1
    #sample average for optimal action percentage
    # and sample average for reward
    R_n_all.append(R_n[n_steps-1])
    R_n_all_c.append(R_n_c[n_steps-1])

print(qstars)
print(Q_a[i_step-2])
print(Q_a_c[i_step-2])
######################################################################################

#plotting the sample average rewards averaged over all episodes for both Q estimate update rules

epsilons = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

f, (ax1) = plt.subplots(1,1)
ax1.set_xlabel("Steps")
ax1.set_ylabel("Average Reward")
ax1.plot(epsilons,R_n_all, epsilons, R_n_all_c)
ax1.legend([r'$\alpha$ = 1/n', r'$\alpha$ = 0.1'])

plt.show()
