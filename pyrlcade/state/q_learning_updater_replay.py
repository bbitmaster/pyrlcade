#this class implements a tabular storage for a Qsa table
import numpy as np
import pyrlcade
from pyrlcade.state.replay_buff import replay_buff

class q_learning_updater_replay(object):
    
    def init(self,update_storage,gamma,use_multiactionnet,
            state_input_size,replay_buf_size,minibatch_size,debug_level):
        self.gamma = gamma
        self.storage = update_storage
        self.use_multiactionnet = use_multiactionnet
        self.debug_level = debug_level
        self.replay_buff = replay_buff()
        self.replay_buff.init(state_input_size,replay_buf_size,debug_level)
        self.state_input_size = state_input_size
        self.replay_buf_size = replay_buf_size
        self.minibatch_size = minibatch_size


    def update(self,alpha,state,action,reward,s_prime,a_prime,qsa_prime_list,is_terminal):
        self.replay_buff.insert(state,action,reward,s_prime,is_terminal)
        minibatch = self.replay_buff.load_minibatch(self.minibatch_size)
        if(minibatch is None):
            return
        s       = minibatch[0]
        a       = minibatch[1]
        r       = minibatch[2]
        s_prime = minibatch[3]
        term    = minibatch[4]
        
        num_actions    = self.storage.num_actions
        minibatch_size = self.minibatch_size
        state_input_size = self.state_input_size
        s_prime_repeat = np.repeat(s_prime,num_actions,axis=1)
        action_possibilities = np.tile(np.arange(num_actions),minibatch_size)

        qsa_prime_allactions = self.storage.load(s_prime_repeat,action_possibilities)
        qsa_prime_reshaped = np.reshape(qsa_prime_allactions,(minibatch_size,num_actions))
        qsa_prime_max = np.max(qsa_prime_reshaped,axis=1)

        #if it is a terminal state, we set the update to r, otherwise we set the update to
        #r + gamma*qsa_prime_max
        update = r + np.invert(term.astype(np.bool))*(self.gamma*qsa_prime_max)

        self.storage.update(alpha,s,a,update)

    def get_qsa_list(self,state):
        if(self.use_multiactionnet):
            return self.storage.loadall(state)
        else:
            #tabular storage does not support loading batches
            if(type(self.storage) is pyrlcade.state.tabular_ram_qsa.tabular_ram_qsa):
                return [self.storage.load(state,i) for i in range(self.storage.num_actions)]
            else:
                actions = np.arange(self.storage.num_actions)
                states = np.repeat(state[:,np.newaxis],self.storage.num_actions,axis=1)
                return list(self.storage.load(states,actions))


if __name__ == '__main__':
    pass
