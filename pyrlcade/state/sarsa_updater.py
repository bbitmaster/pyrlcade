#this class implements a tabular storage for a Qsa table
import numpy as np
import pyrlcade

class sarsa_updater(object):
    
    def init(self,update_storage,gamma,use_multiactionnet,debug_level):
        self.gamma = gamma
        self.storage = update_storage
        self.use_multiactionnet = use_multiactionnet
        self.debug_level = debug_level

    def update(self,alpha,state,action,reward,s_prime,a_prime,qsa_prime_list):
        qsa_prime = qsa_prime_list[a_prime]

        if(self.debug_level >= 4):
            qsa_old = np.copy(self.storage.load(state,action))

        self.storage.update(alpha,state,action, \
            reward + self.gamma*qsa_prime)

        if(self.debug_level >= 4):
            qsa = np.copy(self.storage.load(state,action))
            print("Sarsa Updater: reward: " + str(reward) + (" qsa before: %2.7f" % qsa_old) + (" after: %2.7f" % qsa))

    def get_qsa_list(self,state):
        if(self.use_multiactionnet):
            return self.storage.loadall(state)
        else:
            #tabular storage does not support loading batches
            if(type(self.storage) is pyrlcade.state.tabular_ram_qsa.tabular_ram_qsa):
                return [self.storage.load(state,i) for i in range(self.storage.num_actions)]
            else:
                actions = np.array(range(self.storage.num_actions))
                states = np.repeat(state[:,np.newaxis],self.storage.num_actions,axis=1)
                return list(self.storage.load(states,actions))

if __name__ == '__main__':
    pass
