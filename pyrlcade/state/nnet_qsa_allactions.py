#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
import pyrlcade.misc.cluster_select_func as csf

class nnet_qsa_allactions(object):
    def init(self,state_size,num_actions,p):
        layers = [];
        self.state_size = state_size
        self.num_actions = num_actions

        self.incorrect_target = p['incorrect_target']
        self.correct_target = p['correct_target']
        layers.append(nnet.layer(self.state_size))
        if(p.has_key('num_hidden') and p['num_hidden'] is not None):
            layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
                                     initialization_scheme=p['initialization_scheme'],
                                     initialization_constant=p['initialization_constant'],
                                     dropout=p['dropout'],use_float32=p['use_float32'],
                                     momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
        layers.append(nnet.layer(num_actions,p['activation_function_final'],
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
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        self.net.error = np.zeros(self.net.output.shape,dtype=np.float32)
        self.net.error[action,0] = self.net.output[action,0] - value
        self.net.back_propagate()
        self.net.update_weights()

    def update(self,alpha,state,action,value):
        #update alpha (learning rate)
        for l in self.net.layer:
            l.step_size=alpha

        s = state
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        self.net.error = np.zeros(self.net.output.shape,dtype=np.float32)
        self.net.error[action,0] = -(value - self.net.output[action,0])

        self.net.back_propagate()
        self.net.update_weights()


    def load(self,state,action):
        s =  state
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        return self.net.output[action,0]

    def loadall(self,state):
        s =  state
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        return self.net.output[:,0]


if __name__ == '__main__':
    #TODO: tests?
    pass
