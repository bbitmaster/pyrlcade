import numpy as np

class discrete_bin_storage(object):
    def init(self,mins,maxs,size):
        self.size = np.array(size)
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.data = np.zeros(size)

    def store(self,addr,value):
        s = (np.array(addr) - self.mins)/(self.maxs - self.mins)
        s = s*self.size;
        self.data[tuple(np.rint(s))] = value;

    def load(self,addr):
        s = (np.array(addr) - self.mins)/(self.maxs - self.mins)
        s = s*self.size;
        return self.data[tuple(np.rint(s))]

if __name__ == '__main__':
    mins = [-10,-10,-10,-10]
    maxs = [10,10,10,10]
    size = [20,20,20,20]
    print("mins -10,-10,-10   maxs 10,10,10   size 20,20,20")
    print("initializaing discrete states with 0.0")

    dbs = discrete_bin_storage()
    dbs.init(mins,maxs,size)

    print("storing 2.0 into state 3.0,3.0,3.0,3.0")
    state = [3.0,3.0,3.0,3.0]
    dbs.store(state,2.0);

    state = [3.0,3.0,3.0,3.0]
    val  = dbs.load(state);
    print("state 3.0,3.0,3.0,3.0: " + str(val))

    state = [3.4,3.0,3.0,3.0]
    val  = dbs.load(state);
    print("state 3.4,3.0,3.0,3.0: " + str(val))

    state = [3.6,3.0,3.0,3.0]
    val  = dbs.load(state);
    print("state 3.6,3.0,3.0,3.0: " + str(val))

    state = [4.0,3.0,3.0,3.0]
    val  = dbs.load(state);
    print("state 4.0,3.0,3.0,3.0: " + str(val))

