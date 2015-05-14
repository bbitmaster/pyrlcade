#this class implements a tabular storage for a Qsa table
import numpy as np


class normalization_transformer(object):
    #sets the normalization range
    def init(self,arr_mins,arr_maxs,s_mins,s_maxs):
        self.arr_mins = arr_mins
        self.arr_maxs = arr_maxs
        self.s_mins = s_mins
        self.s_maxs = s_maxs
        self.transform_class=None

    def set_transform_class(self,transform_class):
        self.transform_class = transform_class

    def transform(self,state):
        s = state.astype(np.float32)
        #print('arr_mins: ' + str(self.arr_mins))
        #print('s: ' + str(s))
        s = np.minimum(s,self.arr_maxs)
        s = np.maximum(s,self.arr_mins)
        s = s/(self.arr_maxs)
        s = s*(self.s_maxs-self.s_mins) + self.s_mins
        if(self.transform_class is None):
            return s
        else:
            return self.transform_class.transform(s)

if __name__ == '__main__':
    pass
