#this class implements a tabular storage for a Qsa table
import numpy as np


class pong_ram_extractor(object):
    def __init__(self):
        self.divs = np.array([33,5,1,1,5])
        self.mins = np.array([0x32,0x26,9,7,0x26])
        self.maxs = np.array([0xCD,0xCB,11,13,0xCB])
        
        self.size = self.maxs - self.mins
        self.size = self.size + self.divs
        self.size = self.size/self.divs

        self.arr_mins = (np.zeros(self.size.shape)).astype(np.int64)
        self.arr_maxs = (self.size - np.ones(self.size.shape)).astype(np.int64)
        self.s_mins = self.arr_mins
        self.s_maxs = self.arr_maxs

        self.normalize_func=self.no_normalize

    #a call to this function will turn on normalization and set it to the range specified
    def set_normalization(self,s_mins,s_maxs):
        self.s_mins = s_mins
        self.s_maxs = s_maxs
        self.normalize_func=self.normalize_state

    def extract_state(self,ram):
        #ram values
        #0x31 ball x position 32-CD
        #0x36 ball y position (player's was 26-cB)
        #0x38 ball y velocity -5 to 5
        #0x3A ball x velocity -5 to 5
        #0x3c player y 26-CB
        state = np.array(ram[[0x31,0x36,0x38,0x3A,0x3C]],dtype=np.uint8)
        #these are signed bytes
        state[2] += 10
        state[3] += 10
        state = state - self.mins
        state = state/self.divs
        return self.normalize_func(state)

    def no_normalize(self,state):
        return state

    def normalize_state(self,state):
        s = state.astype(np.float32)
        s = np.minimum(s,self.arr_maxs)
        s = np.maximum(s,self.arr_mins)
        s = s/(self.arr_maxs)
        s = s*(self.s_maxs-self.s_mins) + self.s_mins
        #s = s-0.5
        #s = s*2.25
        return s


if __name__ == '__main__':
    pass
