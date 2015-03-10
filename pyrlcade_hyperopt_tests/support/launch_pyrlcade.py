#!/usr/bin/env python
import subprocess
import os
import sys
import numpy as np
from pyrlcade.runner.main_runner import main_runner

def launch_pyrlcade(paramsfile,params):
    m = main_runner()
    results = m.run(paramsfile,params)
    #compute objective from results
    obj = np.max(results['r_sum_avg_list'])
    argmax = np.argmax(results['r_sum_avg_list'])
    return (obj,argmax)

if __name__ == '__main__':
    p = {}
    print("launching neural network test with 10 episodes of training")
    p['train_episodes'] = 20
    p['skip_saving'] = True
    obj = launch_pyrlcade('../../params/pyrlcade_nnet_params.py',p)
    print("objective function was: " + str(obj))
