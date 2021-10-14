import random
import numpy as np
from matplotlib import pyplot as plt
import timeit

#sample average and constant update step for nonstationary k-arm bandit problem.
# constant step allows for better tracking of nonstationary q* values because it doesn't converge.

k_arms = 10
n_steps = 10000
epsilon = 0.1
runs = 10
qstars_mu = 0
qstars_sd = 1
qstars_mu_add = 0
qstars_sd_add = 0.1
alpha = 0.1

def bandit(A):
    R = np.random.normal(qstars[A],1)
    return R

for r in range(runs):
    print(r)
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
    #tracking number of times the actual optimal action is chosen
    N_Opt_a = 0
    N_Opt_a_c = 0
    #tracking the percentage of times the actual optimal action is chosen
    Opt_a = []
    Opt_a_c = []
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
        #increment number of times the actual optimal action is chosen
        if qstars[A] == qstars[A1]:
            N_Opt_a += 1
        Opt_a.append(N_Opt_a*100/float(i_step))
        if qstars[A_c] == qstars[A1_c]:
            N_Opt_a_c += 1
        Opt_a_c.append(N_Opt_a_c*100/float(i_step))
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
            Q_A_new_c = Q_a_c[i_step-1][A_c] + alpha*(R_c - Q_a_c[i_step - 1][A_c])
            #insert new value estimate
            Q_a[i_step][A] = Q_A_new
            Q_a_c[i_step][A_c] = Q_A_new_c
            #add new average reward (as incremented sample average) for this episode
            R_n[i_step] = R_n[i_step - 1] + 1 / float(i_step) * (R - R_n[i_step - 1])
            R_n_c[i_step] = R_n_c[i_step - 1] + 1 / float(i_step) * (R_c - R_n_c[i_step - 1])
        #increment the step counter
        i_step += 1
    #sample average for optimal action percentage
    # and sample average for reward
    Opt_a_NEW = Opt_a
    R_n_NEW = R_n
    Opt_a_NEW_c = Opt_a_c
    R_n_NEW_c = R_n_c
    if r == 0:
        Opt_a_AV = Opt_a_NEW
        R_n_AV = R_n_NEW
        Opt_a_AV_c = Opt_a_NEW_c
        R_n_AV_c = R_n_NEW_c
    #each step's average reward value is updated pointwise
    # so that at the end of the loop you have one plot
    # with the average rewards (up till that timestep) across all episodes.
    else:
        for n in range(n_steps):
            Opt_a_AV[n] = Opt_a_AV[n] + 1/float(r+1)*(Opt_a_NEW[n]-Opt_a_AV[n])
            Opt_a_AV_c[n] = Opt_a_AV_c[n] + 1 / float(r + 1) * (Opt_a_NEW_c[n] - Opt_a_AV_c[n])
            R_n_AV[n] = R_n_AV[n] + 1/float(r+1)*(R_n_NEW[n] - R_n_AV[n])
            R_n_AV_c[n] = R_n_AV_c[n] + 1 / float(r + 1) * (R_n_NEW_c[n] - R_n_AV_c[n])

Opt_a_AV_conv = Opt_a_AV
R_n_AV_conv = R_n_AV
Opt_a_AV_div_c = Opt_a_AV_c
R_n_AV_div_c = R_n_AV_c
print(qstars)
print(Q_a[i_step-2])
print(Q_a_c[i_step-2])
######################################################################################

#plotting the sample average rewards averaged over all episodes for both Q estimate update rules

timesteps = np.arange(n_steps)
#print(mean_Q_a)

f, (ax1, ax2) = plt.subplots(2,1)
ax1.set_xlabel("Steps")
ax1.set_ylabel("Average Reward")
ax1.plot(timesteps,R_n_AV_conv, timesteps, R_n_AV_div_c)
ax1.legend([r'$\alpha$ = 1/n', r'$\alpha$ = 0.1'])

ax2.plot(timesteps, Opt_a_AV_conv, timesteps, Opt_a_AV_div_c)
ax2.set_xlabel("Steps")
ax2.set_ylabel("% Optimal action ")
ax2.legend([r'$\alpha$ = 1/n', r'$\alpha$ = 0.1'])

plt.show()

