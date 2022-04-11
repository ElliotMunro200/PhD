




if __name__ == "__main__":




# s <- s0
#Choose omega according to an e-soft policy over options pi_Omega(s)
#repeat
#... Choose a according to pi_omega_theta(a|s)
#... Take action a in s, observe s',r
#... 1. Options evaluation:
#... delta <- r - Q_U(s,omega,a)
#... if s' is non-terminal then:
#... ... delta <- delta + gamma*[1-beta_omega_phi(s')]Q_U(s',omega) +
#... ... gamma*beta_omega_phi(s')*max{omega-bar}[Q_Omega(s',omega-bar)]
#... end
#... Q_U(s,omega,a) <- Q_U(s,omega,a) + alpha*delta
#... 2. Options improvement:
#... theta <- theta + alpha_theta*{del/del theta}[log(pi_omega_theta(a|s))]*Q_U(s,omega,a)
#... phi <- phi - alpha_phi*{del/del phi}[beta_omega_phi(s')]*[Q_Omega(s',omega)-V_Omega(s')]
#... if beta_omega_phi terminates in s' then:
#... choose new omega according to e-soft[pi_Omega(s')]
#... s <- s'
#until s' is terminal
