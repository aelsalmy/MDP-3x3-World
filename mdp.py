from constants import *

class MarkovDP:
    def __init__(self , r , disc = 0.99 , size = 3):
        self.size = size
        self.prob = [0.1 , 0.8 , 0.1]
        self.actions = [LEFT , UP , RIGHT , DOWN]
        self.rewards = [[r , -1 , 10] , [-1 , -1 , -1] , [-1 , -1 , -1]]
        self.disc = disc
        self.values = [[0 for i in range(size)] for i in range(size)]

    

