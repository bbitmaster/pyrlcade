#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np

class state_expander_transformer(object):
    def init(self,mins,maxs,new_size,width_scale=1.0):
        layers = [];
        old_size = len(list(mins))
        #self.mins = np.array(mins)
        #self.maxs = np.array(maxs)
        #self.divs = np.array(divs)

        #self.size = self.maxs - self.mins
        #self.size = self.size + self.divs
        #self.size = self.size/self.divs

        #self.arr_mins = (np.zeros(self.size.shape)).astype(np.int64)
        #self.arr_maxs = (self.size - np.ones(self.size.shape)).astype(np.int64)

        self.centroids = np.asarray(((np.random.random((new_size,old_size)) - 0.5) * 2.25),np.float32)
        self.width_scale = width_scale

        #Find the maximum distance between any two rows by using a double for loop
        self.d_max = 0.0
        for i in range(new_size):
            for j in range(new_size):
                d = np.linalg.norm(self.centroids[i,:]-self.centroids[j,:])
                if(d > self.d_max):
                    self.d_max = d
        #This width is suggested in the text:
        #On the Kernel Widths in Radial-Basis Function Networks
        #also in Neural Networks a Comrehensive Foundation by Simon Haykin
        self.sigma = self.d_max/np.sqrt(2*new_size)
        self.sigma = self.sigma**2 #use sigma squared
        self.transform_class = None

    def set_transform_class(self,transform_class):
        self.transform_class = transform_class

    def transform(self,state):
        if(state.ndim == 1):
            #dist = np.linalg.norm(self.centroids - state,axis=1)
            dist = np.sum((self.centroids - state)**2,axis=1)
            #print("dist: " + str(dist))
        else:
            dist = np.sum(self.centroids**2,1)[:,np.newaxis] - 2*np.dot(self.centroids,state) + \
            np.sum(state**2,0)[np.newaxis,:]
            #print("dist: " + str(dist))
        
        dist_inv = np.exp(-dist/(2*self.sigma*self.width_scale))
        if(self.transform_class is None):
            return dist_inv
        else:
            return self.transform_class.transform(dist_inv)

if __name__ == '__main__':
    mins = np.zeros(5)
    maxs = np.ones(5)
    s = state_expander()
    s.init(mins,maxs,10)

    print("Max Distance: " + str(s.d_max))
    print("width: " + str(s.sigma))

    a = np.random.random(5)
    print("random vector: " + str(a))
    a_t = s.transform(a)
    print("  transformed: " + str(a_t))

    for i in range(6):
        print("Testing vector closest to " + str(i))
        a = s.centroids[i,:]
        print("            a: " + str(a))
        a_t = s.transform(a)
        print("  transformed: " + str(a_t))
    



