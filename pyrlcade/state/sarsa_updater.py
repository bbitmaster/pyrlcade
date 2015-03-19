#this class implements a tabular storage for a Qsa table
import numpy as np

class sarsa_updater(object):
    
    def init(self,update_storage,gamma):
        self.gamma = gamma
        self.storage = update_storage

    def update(self,state,action,reward,s_prime,a_prime,qsa_prime_list):
        qsa_prime = qsa_prime_list[a_prime]

        self.storage.update(state,action, \
            reward + self.gamma*qsa_prime)

    def get_qsa_list(self,state):
        qsa_list = [self.storage.load(state,i) for i in range(self.storage.num_actions)]
        return qsa_list

if __name__ == '__main__':
    pass
