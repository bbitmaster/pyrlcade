#this class implements a tabular storage for a Qsa table
import numpy as np

class tabular_ram_qsa(object):
    def init(self,mins,maxs,divs,num_actions,alpha):
        self.alpha = alpha
        self.num_actions = num_actions
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.size = self.maxs - self.mins
        self.divs = np.array(divs)
        self.size = self.size + self.divs
        self.size = self.size/self.divs

        self.arr_mins = (np.zeros(self.size.shape)).astype(np.int64)
        self.arr_maxs = (self.size - np.ones(self.size.shape)).astype(np.int64)
        self.data = []
        for a in range(self.num_actions):
            self.data.append(-np.random.random(self.size)/100.0)

    def store(self,state,action,value):
        s =  state - self.mins
        s /= self.divs
        s = np.minimum(s,self.arr_maxs)
        s = np.maximum(s,self.arr_mins)
        self.data[action][tuple(s)] = value

    def update(self,state,action,value):
        s =  state - self.mins
        s /= self.divs
        s = np.minimum(s,self.arr_maxs)
        s = np.maximum(s,self.arr_mins)
        d = self.data[action][tuple(s)]
        self.data[action][tuple(s)] = d + self.alpha*(value - d)
        #above can also be expressed as:
        #d*(1 - self.alpha) + self.alpha*value

    def load(self,state,action):
        s =  state - self.mins
        s /= self.divs
        s = np.minimum(s,self.arr_maxs)
        s = np.maximum(s,self.arr_mins)
        #print("divs: " + str(self.divs))
        #print("s: " + str(s))
        #print("self.data[action].shape: " + str(self.data[action].shape))
        return self.data[action][tuple(s)]

if __name__ == '__main__':
    mins = [10,20,30,40]
    maxs = [100,100,100,100]
    divs = [1,1,4,1]

    print("mins 10,20,30,40   maxs 100,100,100,100 actions: 4")
    print("initializaing discrete states with 0.0")

    qsa = tabular_ram_qsa()
    qsa.init(mins,maxs,divs,4)

    print("storing 2.0 into state 11,21,31,41 action:0")
    state = [99,21,31,41]
    state = np.array(state,dtype=np.uint32)
    qsa.store(state,0,2.0);

    print("updating 1.0 into state 11,21,31,41 action:0")
    state = [99,21,31,41]
    state = np.array(state,dtype=np.uint32)
    qsa.update(state,0,1.0);

    state = [99,21,31,41]
    val  = qsa.load(state,0);
    print("state 99,21,31,41 action 0: " + str(val))

    state = [99,21,31,41]
    val  = qsa.load(state,1);
    print("state 99,21,31,41 action 1: " + str(val))

    state = [99,21,33,41]
    val  = qsa.load(state,0);
    print("state 99,21,33,41 action 0: " + str(val))

    state = [99,21,34,41]
    val  = qsa.load(state,0);
    print("state 99,21,34,41 action 0: " + str(val))

#    val = 2.0
#    val  = [qsa.load(state,i) for i in range(4)]
#    print("state 4.0,3.0,3.0,3.0 action 0: " + str(val))
#    val[0] = 2.0
#    print("state 4.0,3.0,3.0,3.0 action 0: " + str(val))
#    val  = [qsa.load(state,i) for i in range(4)]
#    print("state 4.0,3.0,3.0,3.0 action 0: " + str(val))
