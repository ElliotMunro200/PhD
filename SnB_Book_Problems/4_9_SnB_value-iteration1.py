import numpy as np
import matplotlib.pyplot as plt

#training the policy through value iteration
def train(ph=0.4, Theta=0.000001): #initialising parameters
    #initialising V_s and pi
    V = [0]*100
    for i in range(0, 100):
        V[i] = np.random.random() * 1000
    V[0] = 0
    pi = [0]*100
    counter = 1
    while True:
        Delta = 0
        for s in range(1, 100):  # for each state
            old_v = V[s]
            v = [0] * 51 #using temporary variable v to compute max and argmax values more easily for a given state
            for a in range(1, min(s, 100 - s) + 1): #restricting values of a to 1...min(s,100-s)
                v[a] = 0
                if a + s < 100:
                    v[a] += ph * (0 + V[s + a])
                    v[a] += (1 - ph) * (0 + V[s - a])
                elif a + s == 100:
                    v[a] += ph
                    v[a] += (1 - ph) * (0 + V[s - a])
            op_a = np.argmax(v)
            pi[s] = op_a
            V[s] = v[op_a]
            Delta = max(Delta, abs(old_v - V[s]))
        counter += 1
        if counter % 1000 == 0:
            print("train loop" + str(counter))
            print("Delta =" + str(Delta))
        if Delta < Theta:
            break
    return [V[1:100], pi[1:100]]

def draw(S,pi,ph):
    plt.figure()
    plt.bar(S, pi)
    plt.xlabel("Capital")
    plt.ylabel("Stakes for Ph="+str(ph))
    plt.show()

if __name__ == "__main__":
    [V1, pi1] = train(ph=0.4)
    [V2, pi2] = train(ph=0.25)
    [V3, pi3] = train(ph=0.55)
    [V4, pi4] = train(ph=0.50)
    S = np.linspace(1, 99, num=99, endpoint=True)
    plt.figure()
    plt.plot(S, V1)
    plt.plot(S, V2)
    plt.plot(S, V3)
    plt.plot(S, V4)
    plt.legend(["0.4", "0.25", "0.55", "0.5"])
    plt.show()

    draw(S, pi1, 0.4)
    draw(S, pi2, 0.25)
    draw(S, pi3, 0.55)
    draw(S, pi4, 0.50)


