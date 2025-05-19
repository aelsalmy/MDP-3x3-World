from mdp import *

mdp = MarkovDP(100)

mdp.policy_iteration()

mdp.print_value_func()