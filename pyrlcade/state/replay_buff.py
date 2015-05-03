#This class implements the actual buffer used for a replay buffer
import numpy as np

class replay_buff(object):
    def init(self,state_size,max_buf_size,debug_level=1):
        self.s = np.zeros((state_size,max_buf_size),dtype=np.float32)
        self.s2 = np.zeros((state_size,max_buf_size),dtype=np.float32)
        self.a = np.zeros((max_buf_size),dtype=np.float32)
        self.r = np.zeros((max_buf_size),dtype=np.float32)
        self.term = np.zeros((max_buf_size),dtype=np.int)
        self.max_buf_size = max_buf_size
        self.buf_size = 0
        self.shuffled_size=0
        self.load_index=0
        self.debug_level = debug_level

    def insert(self,s,a,r,s2,term):
        if(self.buf_size < self.max_buf_size):
            index = self.buf_size
            self.buf_size+=1
        else:
            index = np.random.randint(self.max_buf_size)
        self.s[:,index] = s
        self.a[index] = a
        self.r[index] = r
        self.s2[:,index] = s2
        self.term[index] = term


    def load_minibatch(self,minibatch_size):
        #print("buf_size: " + str(self.buf_size) + " load_index: " + str(self.load_index) + " shuffled_size: " + str(self.shuffled_size))
        #minibatch size must be at least buf size
        if(self.buf_size < minibatch_size):
            return
        #reset load index to 0 if it goes beyond what has been stored
        if(self.buf_size < self.load_index+minibatch_size):
            if(self.debug_level >= 2):
                print("shuffling r_buff load_index: " + str(self.load_index) + " buf_size: " + str(self.buf_size) + " self.shuffled_size: " + str(self.shuffled_size) + " 1")
            self.shuffle()
            self.load_index = 0

        if(self.load_index > self.shuffled_size):
            if(self.debug_level >= 2):
                print("shuffling r_buff load_index: " + str(self.load_index) + " buf_size: " + str(self.buf_size) + " self.shuffled_size: " + str(self.shuffled_size) + " 2")
            self.shuffle()
            self.load_index = 0
        s = self.s[:,self.load_index:(self.load_index+minibatch_size)]
        a = self.a[self.load_index:(self.load_index+minibatch_size)]
        r = self.r[self.load_index:(self.load_index+minibatch_size)]
        s2 = self.s2[:,self.load_index:(self.load_index+minibatch_size)]
        term = self.term[self.load_index:(self.load_index+minibatch_size)]
        self.load_index += minibatch_size
        return (s,a,r,s2,term)

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.s.T[0:self.buf_size])
        np.random.set_state(rng_state)
        np.random.shuffle(self.a[0:self.buf_size])
        np.random.set_state(rng_state)
        np.random.shuffle(self.r[0:self.buf_size])
        np.random.set_state(rng_state)
        np.random.shuffle(self.s2.T[0:self.buf_size])
        np.random.set_state(rng_state)
        np.random.shuffle(self.term[0:self.buf_size])
        self.shuffled_size = self.buf_size


if __name__ == '__main__':
    import sys
    if(len(sys.argv) > 3):
        state_size = int(sys.argv[1])
        max_buf_size = int(sys.argv[2])
        minibatch_size = int(sys.argv[3])
    else:
        state_size = 4
        max_buf_size = 32
        minibatch_size = 4
    print("state size: " + str(state_size))
    print("buffer size: " + str(max_buf_size))
    print("minibatch size: " + str(minibatch_size))
    
    rbuf = replay_buff()
    rbuf.init(state_size,max_buf_size)


    #generate some inputs
    autoincrement = 0
    for i in range(max_buf_size*2):
        s = np.tile(autoincrement,(state_size))
        a = autoincrement
        r = autoincrement
        s2 = np.tile(autoincrement,(state_size))
        term = np.random.randint(2)
        rbuf.insert(s,a,r,s2,term)
        data = rbuf.load_minibatch(minibatch_size)
        print("iteration: " +str(autoincrement) + " data: " +str(data))
        autoincrement += 1

