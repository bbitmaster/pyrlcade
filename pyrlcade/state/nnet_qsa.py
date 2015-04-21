#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
import pyrlcade.misc.cluster_select_func as csf

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
                                     dropout=p['dropout'],use_float32=p['use_float32'],
                                     momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
        layers.append(nnet.layer(1,p['activation_function_final'],
                                 initialization_scheme=p['initialization_scheme_final'],
                                 initialization_constant=p['initialization_constant_final'],
                                 use_float32=p['use_float32'],
                                 momentum=p['momentum'],step_size=p['learning_rate']))
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
        #print('nnet state: ' + str(s.transpose()))
        self.net.input = s
        self.net.feed_forward()
        #value = r + gamma*qsa_max
        #thus, net.error = -(r + gamma*qsa_max - qsa)
        #with learning rate it becomes: -alpha(r + gamma*qsa_max - qsa)
        #then, the target becomes the temporal difference update of:
        #qsa - alpha(r + gamma*qsa_max - qsa)

        #err = (output - target)
        #err = qsa - alpha(r + gamma*qsa_max - qsa)
        #err = -alpha(r + gamma*qsa_max - qsa)
        #err = -(r + gamma*qsa_max - qsa)
        #err = -(value - qsa)
        #err = -(value - net.output)
        #print('update output: ' + str(net.output))
        self.net.error = -(value - self.net.output)

        self.net.back_propagate()
        self.net.update_weights()

    def load(self,state,action):
        #if a single instance is passes, make it a vector
        if(type(action) is int):
            action = np.array([action])
            state = state[:,np.newaxis]
        #build matrix of one hot actions
        action_list = np.ones((self.num_actions,action.shape[0]))*self.incorrect_target
        for i in range(action.shape[0]):
            action_list[action[i],i] = self.correct_target
        #append state matrix to action matrix
        s = np.append(state,action_list,axis=0)
        self.net.input = s
        self.net.feed_forward()
        return self.net.output[0]

if __name__ == '__main__':
    #TODO: tests?
    pass
