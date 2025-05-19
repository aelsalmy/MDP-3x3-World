from constants import *
import copy
import math
import random

class MarkovDP:
    def __init__(self , r , disc = 0.99 , size = 3):
        self.size = size
        self.prob = [0.1 , 0.8 , 0.1]
        self.actions = [LEFT , UP , RIGHT , DOWN]
        self.rewards = [[r , -1 , 10] , [-1 , -1 , -1] , [-1 , -1 , -1]]
        self.disc = disc
        self.values = [[0 for i in range(size)] for i in range(size)]
        self.action_effect = {
            DOWN: (1 , 0),
            UP: (-1 , 0),
            LEFT: (0 , -1),
            RIGHT: (0 , 1)
        }
        self.policy = [ self.actions.copy() for i in range(self.size * self.size) ]
        self.policy[2] = []     #Terminal node has no possible actions

    #function to get possible actions of a specific state
    def get_possible_actions(self , i , j):

        actions = self.policy[i * self.size + j]
        
        for a in actions:

            x , y = self.action_effect[a]
            t_x , t_y = x + i , y + j

            if t_x >= self.size or t_x < 0 or t_y < 0 or t_y >= self.size:
                actions.remove(a)

        return actions
    
    def get_general_possible_actions(self, i, j):
        valid_actions = []

        for a in self.actions:
            dx, dy = self.action_effect[a]
            nx, ny = i + dx, j + dy

            if 0 <= nx < self.size and 0 <= ny < self.size:
                valid_actions.append(a)

        return valid_actions

    def get_action_rot(self , a):

        idx = self.actions.index(a)

        return[
            self.actions[(idx - 1) % 4],
            a,
            self.actions[(idx + 1) % 4]
        ]

    def get_prev_value(self , i , j):
        if i >= self.size or i < 0 or j >= self.size or j < 0:
            return 0
        else:
            return self.values[i][j]
    
    #get value acc to specific action g(s,a)
    def get_action_value(self , i , j , a):
        temp = self.rewards[i][j]

        ind = -1

        rot = self.get_action_rot(a)

        for probability , direction in zip(self.prob , rot):

            x , y = self.action_effect[direction]
            temp += self.disc * probability * self.get_prev_value(i + x , j + y)

            ind += 1

        return temp

    #algorithm to apply policy eval to get value function of iteration
    def policy_eval(self):

        while True:

            change = 0
            new_value = copy.deepcopy(self.values)

            #iterate through each state of world
            for i in range(self.size):
                for j in range(self.size):

                    actions = self.get_possible_actions(i , j)
                    
                    if len(actions) != 0:
                        prob_of_action = 1 / (len(actions))

                    else:
                        self.values[i][j] = self.rewards[i][j]
                        continue

                    new_value[i][j] = prob_of_action * sum(self.get_action_value(i , j , a) for a in actions)

                    change = max(change , abs(new_value[i][j] - self.values[i][j]))

            self.values = new_value
            if change < THRESHOLD:
                break
    
    def generate_random_policy(self):
        self.policy = []

        for i in range(self.size):
            for j in range(self.size):
                if i == 0 and j == 2:  # terminal state
                    self.policy.append([])
                    continue

                valid_actions = self.get_general_possible_actions(i , j)

                chosen_action = random.choice(valid_actions)
                self.policy.append([chosen_action])

    def policy_iteration(self):

        self.generate_random_policy()

        print(self.policy)
        
        policy_updated = True
        iterations = 0

        while policy_updated:
            policy_updated = False

            iterations += 1
            print(f"iteration: {iterations}")

            self.policy_eval()

            for i in range(self.size):
                for j in range(self.size):

                    if i == 0 and j == 2:       #terminal state
                        continue

                    state = i * self.size + j
                    state_policy = self.policy[state]
                    
                    max_val = -1 * math.inf
                    optimal_action = -1

                    for a in self.get_general_possible_actions(i , j):

                        val = self.get_action_value(i , j , a)

                        if max_val < val:
                            max_val = val
                            optimal_action = a
                    
                    if [optimal_action] != state_policy:
                        self.policy[state] = [optimal_action]
                        policy_updated = True
        
        print("Policy Iteration Done")
        print(self.policy)

    def print_value_func(self):

        print("\nValue Function:")
        for row in self.values:
            print(" | ".join(f"{v:6.2f}" for v in row))
        print()                    



