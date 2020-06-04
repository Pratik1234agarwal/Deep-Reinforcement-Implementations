import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.esp = 0.6
        self.alpha = 0.02
        self.iterations=1
        self.gamma = 0.9

    def select_action(self, state):
        self.iterations+=1
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        prob = self.get_prob(state)
        return np.random.choice(np.arange(self.nA),p=prob)
    def get_prob(self,state):
        epsilon = 1.0/self.iterations
        prob = np.ones(self.nA)*(epsilon/self.nA)
        val = np.argmax(self.Q[state])
        prob[val] = (1-epsilon) + epsilon/self.nA
        return prob
    def get_Q_value(self,state):
        prob = self.get_prob(state)
        return sum(prob*self.Q[state])
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            self.Q[state][action] += self.alpha*(reward+ (self.gamma*self.get_Q_value(next_state)) - self.Q[state][action])
            #self.Q[state][action] += self.alpha*(reward+ (self.gamma*max(self.Q[next_state])) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha*(reward-self.Q[state][action])
                          
                          