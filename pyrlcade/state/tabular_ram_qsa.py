#this class implements a tabular storage for a Qsa table
import numpy as np

class tabular_ram_qsa(object):
    def init(self,mins,maxs,num_actions,p):
        self.num_actions = num_actions
        self.mins = np.copy(np.array(mins)).astype(np.int64)
        self.maxs = np.copy(np.array(maxs)).astype(np.int64)
        self.size = self.maxs - self.mins

        #self.arr_mins = (np.zeros(self.size.shape)).astype(np.int64)
        #self.arr_maxs = (self.size - np.ones(self.size.shape)).astype(np.int64)
        self.data = []
        for a in range(self.num_actions):
            self.data.append(-np.random.random(self.size + np.ones(self.size.shape))/100.0)

    def store(self,state,action,value):
        s =  state - self.mins
        s = np.minimum(s,self.maxs)
        s = np.maximum(s,self.mins)
        self.data[action][tuple(s)] = value

    def update(self,alpha,state,action,value):
        s =  state - self.mins
        s = np.minimum(s,self.maxs)
        s = np.maximum(s,self.mins)
        d = self.data[action][tuple(s)]
        self.data[action][tuple(s)] = d + alpha*(value - d)
        #above can also be expressed as:
        #d*(1 - self.alpha) + self.alpha*value

    def load(self,state,action):
        s =  state - self.mins
        s = np.minimum(s,self.maxs)
        s = np.maximum(s,self.mins)
        #print("s: " + str(s))
        #print("self.data[action].shape: " + str(self.data[action].shape))
        return self.data[action][tuple(s)]

if __name__ == '__main__':
    mins = [10,20,30,40]
    maxs = [100,100,100,100]

    print("mins 10,20,30,40   maxs 100,100,100,100 actions: 4")
    print("initializaing discrete states with 0.0")

    qsa = tabular_ram_qsa()
    qsa.init(mins,maxs,4)

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
