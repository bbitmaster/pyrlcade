#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
import pyrlcade.misc.cluster_select_func as csf
from copy import deepcopy

class nnet_qsa(object):
    def init(self,state_size,num_actions,p):
        layers = [];
        self.state_size = state_size
        self.num_actions = num_actions
        #self.mins = np.array(mins)
        #self.maxs = np.array(maxs)
        #self.divs = np.array(divs)

        #self.size = self.maxs - self.mins
        #self.size = self.size + self.divs
        #self.size = self.size/self.divs

        #self.arr_mins = (np.zeros(self.size.shape)).astype(np.int64)
        #self.arr_maxs = (self.size - np.ones(self.size.shape)).astype(np.int64)


        self.incorrect_target = p['incorrect_target']
        self.correct_target = p['correct_target']
        layers.append(nnet.layer(self.state_size + self.num_actions))
        if(p.has_key('num_hidden') and p['num_hidden'] is not None):
            layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
                                     initialization_scheme=p['initialization_scheme'],
                                     initialization_constant=p['initialization_constant'],
                                     dropout=p.get('dropout',None),use_float32=p['use_float32'],
                                     momentum=p['momentum'],maxnorm=p.get('maxnorm',None),step_size=p['learning_rate'],rms_prop_rate=p.get('rms_prop_rate',None)))

        layers.append(nnet.layer(1,p['activation_function_final'],
                                 initialization_scheme=p['initialization_scheme_final'],
                                 initialization_constant=p['initialization_constant_final'],
                                 use_float32=p['use_float32'],
                                 momentum=p['momentum'],step_size=p['learning_rate'],rms_prop_rate=p.get('rms_prop_rate',None)))
        self.net = nnet.net(layers)

        self.do_neuron_clustering=False #by default
        if(p.has_key('cluster_func') and p['cluster_func'] is not None):
            self.net.layer[0].centroids = np.asarray(((np.random.random((self.net.layer[0].weights.shape)) - 0.5) * 2.5),np.float32)
            #make the centroid bias input match the bias data of 1.0
            self.net.layer[0].centroids[:,-1] = 1.0
            #print(str(self.net.layer[0].centroids.shape))
            #print(str(self.net.layer[0].centroids))
            self.net.layer[0].select_func = csf.select_names[p['cluster_func']]
            print('cluster_func: ' + str(csf.select_names[p['cluster_func']]))
            self.net.layer[0].centroid_speed = p['cluster_speed']
            self.net.layer[0].num_selected = p['clusters_selected']
            self.do_neuron_clustering=True #set a flag to indicate neuron clustering
            if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
                self.net.layer[0].do_cosinedistance = True
                print('cosine set to true')
        self.max_update = 0.0
        self.grad_clip = p.get('grad_clip',None)

    def store(self,state,action,value):
        s = state
        #s /= self.divs
        #s = np.minimum(s,self.arr_maxs)
        #s = np.maximum(s,self.arr_mins)
        #s = s/(self.arr_maxs)
        #s = s-0.5
        #s = s*2.25
        action_list = np.ones((1,self.num_actions))*self.incorrect_target
        action_list[0,action] = self.correct_target
        s = np.append(s,action_list)[:,np.newaxis]
        self.net.input = s
        self.net.feed_forward()
        self.net.error = self.net.output - value
        self.net.back_propagate()
        self.net.update_weights()

    def update(self,alpha,state,action,value):
        #update alpha (learning rate)
        for l in self.net.layer:
            l.step_size=alpha

        #if a single instance is passed, make it a vector
        if(type(action) is not np.ndarray):
            action = np.array([action])
            state = state[:,np.newaxis]

        #build matrix of one hot actions
        action_list = np.ones((self.num_actions,action.shape[0]),dtype=np.float32)*self.incorrect_target
        for i in range(action.shape[0]):
            action_list[action[i],i] = self.correct_target

        #append state matrix to action matrix
        s = np.append(state,action_list,axis=0)
        
        self.net.input = s
        self.net.feed_forward()

        self.net.error = -(value - self.net.output)
        #print the largest update so far
        if(np.max(np.abs(self.net.error)) > np.abs(self.max_update)):
            self.max_update = np.max(np.abs(self.net.error))
        #gradient clipping
        if(self.grad_clip is not None):
            self.net.error[self.net.error > self.grad_clip] = self.grad_clip
            self.net.error[self.net.error < -self.grad_clip] = -self.grad_clip
        self.net.back_propagate()
        self.net.update_weights()

        #value = r + gamma*qsa_max
        #thus, net.error = -(r + gamma*qsa_max - qsa)
        #with learning rate it becomes: -alpha(r + gamma*qsa_max - qsa)
        #then, the target becomes the temporal difference update of:
        #qsa - alpha(r + gamma*qsa_max - qsa)

    def load(self,state,action):
        #if a single instance is passed, make it a vector
        if(type(action) is not np.ndarray):
            action = np.array([action])
            state = state[:,np.newaxis]
        #build matrix of one hot actions
        action_list = np.ones((self.num_actions,action.shape[0]),dtype=np.float32)*self.incorrect_target
        for i in range(action.shape[0]):
            action_list[action[i],i] = self.correct_target
        #append state matrix to action matrix
        s = np.append(state,action_list,axis=0)

        if(hasattr(self,'frozen_net')):
            net = self.frozen_net
        else:
            net = self.net
        net.input = s
        net.feed_forward()
        return net.output[0]

    def create_frozen_qsa_storage(self):
        self.frozen_net = deepcopy(self.net)

    def delete_frozen_qsa_storage(self):
        del self.frozen_qsa_net

if __name__ == '__main__':
    #TODO: tests?
    pass
