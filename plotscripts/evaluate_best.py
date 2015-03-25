#!/usr/bin/env python
import subprocess
import os
import h5py as h5
import numpy as np
import sys
from operator import itemgetter
from pyrlcade.misc.save_h5py import save_results,load_results

def evaluate_params(fname):
    results = load_results(fname)
    if(results['r_sum_avg_list'].size == 0):
        return None
    results['obj'] = np.max(np.array(results['r_sum_avg_list']))
    results['argmax'] = np.argmax(np.array(results['r_sum_avg_list']))
    return results

def print_sorted(p_list):
    new_p = reversed(sorted(p_list,key=itemgetter('obj'),reverse=True))

    print("")
    for p in new_p:
        #print(str(p))
        print("Filename: " + str(p['f_name']))
        if(p.has_key('os')):
            print("machine: " + str(p['os']))
        print("obj: " + str(p['obj']))
        print("argmax: " + str(p['argmax']))
        print("episode: " + str(p['episode']))
        print("Parameters: ")
        for k,v in sorted(p['parameters'].items(),key=lambda x: x[0]):
            #print(str(param))
            #for k,v in param.items():
            print("\t" + str(k) + " : " + str(v))

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("usage: evaluate <path>")
        sys.exit()
    path=sys.argv[1]
    p_list = []
    print("path:" + str(path))
    for root, dirs, files in os.walk(path):
        for f in files:
            path = os.path.join(root,f)
            #print('loading... ' + path)
            sys.stdout.write('.')
            sys.stdout.flush()
            p = evaluate_params(path)
            if(p is None):
                continue
            p['f_name'] = path
            p_list.append(p)
    print_sorted(p_list)

