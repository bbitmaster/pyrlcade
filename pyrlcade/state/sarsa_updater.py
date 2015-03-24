#this class implements a tabular storage for a Qsa table
import numpy as np

class sarsa_updater(object):
    
    def init(self,update_storage,gamma,use_multiactionnet):
        self.gamma = gamma
        self.storage = update_storage
        self.use_multiactionnet = use_multiactionnet

    def update(self,alpha,state,action,reward,s_prime,a_prime,qsa_prime_list):
        qsa_prime = qsa_prime_list[a_prime]

        self.storage.update(alpha,state,action, \
            reward + self.gamma*qsa_prime)

    def get_qsa_list(self,state):
        if(self.use_multiactionnet):
            return self.storage.loadall(state)
        else:
            return [self.storage.load(state,i) for i in range(self.storage.num_actions)]

if __name__ == '__main__':
    pass
