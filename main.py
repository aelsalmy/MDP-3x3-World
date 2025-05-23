from mdp import *

mdp = MarkovDP(r=3)

mdp.value_iteration()

mdp.print_value_func()