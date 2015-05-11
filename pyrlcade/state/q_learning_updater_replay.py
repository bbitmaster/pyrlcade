#this class implements a tabular storage for a Qsa table
import numpy as np
import pyrlcade
from pyrlcade.state.replay_buff import replay_buff

class q_learning_updater_replay(object):
    
    def init(self,update_storage,gamma,use_multiactionnet,
            state_input_size,replay_buf_size,minibatch_size,debug_level,update_freeze_rate):
        self.gamma = gamma
        self.storage = update_storage
        self.use_multiactionnet = use_multiactionnet
        self.debug_level = debug_level
        self.replay_buff = replay_buff()
        self.replay_buff.init(state_input_size,replay_buf_size,debug_level)
        self.state_input_size = state_input_size
        self.replay_buf_size = replay_buf_size
        self.minibatch_size = minibatch_size
        self.update_freeze_rate = update_freeze_rate
        self.update_count = 0


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

        #print("s dtype: " + str(s.dtype))
        #print("a dtype: " + str(a.dtype))
        #print("r dtype: " + str(r.dtype))
        #print("s_prime dtype: " + str(s_prime.dtype))
        #print("term dtype: " + str(term.dtype))
        #print("s_prime_repeat dtype: " + str(s_prime_repeat.dtype))
        #print("action_possibilities dtype: " + str(action_possibilities.dtype))
        #print("qsa_prime_allactions dtype: " + str(qsa_prime_allactions.dtype))
        #print("qsa_prime_max dtype: " + str(qsa_prime_max.dtype))
        #print("update dtype: " + str(update.dtype))
        self.storage.update(alpha,s,a,update)

        if(self.update_freeze_rate is not None):
            if(type(self.storage) is pyrlcade.state.nnet_qsa.nnet_qsa):
                if((self.update_count % self.update_freeze_rate) == 0):
                    self.storage.create_frozen_qsa_storage()
                    if(self.debug_level >= 2):
                        print('freezing net...')
                self.update_count +=1


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
