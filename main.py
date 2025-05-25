from mdp import *

r = [100 , 3 , 1 , 0 , -3]
#r = [100 , 0]

for reward in r:
    print(f"Optimal policy for r = {reward} applying Value Iteration:\n")
    
    mdp = MarkovDP(r = reward)

    mdp.value_iteration()

    mdp.print_value_func()

    print("--------------------------------------")
    print(f"Optimal policy for r = {reward} applying Policy Iteration:\n")

    mdp2 = MarkovDP(r = reward)

    mdp2.policy_iteration()

    mdp2.print_value_func()

    print("--------------------------------------")
