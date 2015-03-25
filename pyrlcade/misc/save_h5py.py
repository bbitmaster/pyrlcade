#!/usr/bin/env python
import sys
import numpy as np
import h5py as h5

def save_results(filename,results):
    f_handle = h5.File(filename,'w')
    for k,v in results.iteritems():
        if(type(v) in (int,float,str,np.ndarray)):
            #print("k:v  " + str(k) + ":" + str(v))
            f_handle[k] = np.array(v)
        #the parameters are stored as a dict within the results dict, this supports saving them
        if(type(v) == dict):
            p_group = f_handle.create_group(k);
            for param in v.iteritems():
                #only save the ones that have a data type that is supported
                if(type(param[1]) in (bool,int,float,str,np.ndarray)):
                    p_group[param[0]] = param[1];
    f_handle.close();

def load_results(filename):
    results = {}
    try:
        f_handle = h5.File(filename,'r')
    except IOError:
        print("Error Opening: " + str(filename))
        raise IOError
    for k,v in f_handle.iteritems():
        if(type(v) == h5._hl.dataset.Dataset):
            results[str(k)] = np.array(v)
            #is this a 0-dim array with one element? if so, remove it
            if(len(results[str(k)].shape) == 0):
                results[str(k)] = results[str(k)].item()
        #save any groups as dictionaries
        if(type(v) == h5._hl.group.Group):
            group = dict(v)
            p = {}
            for g_k,g_v in group.iteritems():
                p[str(g_k)] = g_v.value
            results[str(k)] = p
    f_handle.close();
    return results

if __name__ == '__main__':
    save_data = {}
    save_data['test1'] = 1.2
    save_data['test2'] = 3
    save_data['test3'] = np.random.random((3,4))

    p = {}
    p['some_parameter1'] = 'blah'
    p['some_parameter2'] = 12
    p['some_parameter3'] = 0.99
    save_data['parameters'] = p

    print("save data: " + str(save_data))
    save_results('tmp_file',save_data)

    load_data = load_results('tmp_file')

    print("")
    print("load data: " + str(load_data))

