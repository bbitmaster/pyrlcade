#!/usr/bin/env python
import sys
import math
import time
from hyperopt import fmin, tpe, hp, mix, rand, STATUS_OK, STATUS_FAIL
from functools import partial
import hyperopt
from hyperopt.mongoexp import MongoTrials
from pyrlcade_hyperopt_tests.support.launch_pyrlcade import launch_pyrlcade


space = hp.choice('main_choice',[{
    'activation_function_final' : hp.choice('a_function_final',['tanh','squash','linear']),
    'num_hidden'                : None,
    'rbf_transform_size'        : hp.choice('rbf_t_size',[10,20,50,100,500,1000]),
    'rbf_width_scale'           : hp.choice('rbf_w_scale',[1.0,10.0,25.0,50.0,100.0,150.0,200.0,500.0]),
    'learning_rate'             : hp.choice('l_rate',[0.1,0.01,.001,.0001]),
    'learning_rate_decay'       : hp.choice('l_rate_decay',[0.9995,0.9998,0.9999,1.0]),
    'learning_rate_min'         : hp.choice('l_rate_min',[0.0001,0.00001]),
    'gamma'                     : hp.choice('gamm',[0.9,0.95,0.97,0.98,0.99,0.995,0.999]),
    'action_type'               : 'e_greedy',
    'cluster_func'              : None,
    'decay_type'                : 'geometric',
    'nnet_use_combined_actions' : False,
    'epsilon'                   : hp.choice('eps',[0.99,0.50,0.20]),
    'epsilon_decay'             : hp.choice('eps_decay',[0.9998,0.9997,0.9995,0.9992,0.999]),
    'epsilon_min'               : hp.choice('eps_min',[0.05,0.02,0.01,0.005]),
    'reward_multiplier'         : hp.choice('r_mult',[1.0,0.75,0.5,0.2,0.10,0.01]),
    'rl_algo'                   : hp.choice('r_algo',['sarsa','q_learning'])
    }])

def objective(space):
    #This function will be unpickled by hyperopt and we need to reimport everythinng for it to work
    import time
    from hyperopt import STATUS_OK, STATUS_FAIL
    import random
    import sys
    import math
    import os
    from pyrlcade_hyperopt_tests.support.launch_pyrlcade import launch_pyrlcade
    import pyrlcade_hyperopt_tests

    os.path.dirname(pyrlcade_hyperopt_tests.__file__)

    #we cheat to get the directory of the parameters file. we find the directory of the hyperopt_tests import
    cwd = os.path.dirname(pyrlcade_hyperopt_tests.__file__)
    
    paramsfile_relative = '../params/pyrlcade_rbf_linear_params.py'
    paramsfile = os.path.abspath(os.path.join(cwd,paramsfile_relative))

    params = space

    #since hyperopt is ran from somewhere else, set the results directory correctly for saving results
    params['results_dir'] = os.path.abspath(os.path.join(cwd,'../results/')) + '/'
    params['rom_file'] = os.path.join(os.path.abspath(os.path.join(cwd,'../roms/')),'pong.bin')

    #set any parameteers here to override the parameters in the params file
    params['simname'] = 'pyrlcade_rbftest'
    params['save_interval']=60

    #Give each run a random unique identifier for the version. This allows us to locate any saved results via a signature in the filename
    rnd_str = '_'
    for i in range(12):
        rnd_str += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    params['version'] = rnd_str
    t = time.time()
    
    #TODO? Maybe catch exceptions and set status to failed if the occur?
    #try:
    (obj,argmin) = launch_pyrlcade(paramsfile,params)

    print('obj: ' + str(obj) + ' argmin: ' + str(argmin))
    return {
        'loss' : obj,
        'argmin' : str(argmin),
        'status' : STATUS_OK,
        'eval_time': str(time.time() - t),
        'rnd_str' : rnd_str
        #any additional logging goes here
#        'attachments':
#            {'stdout' : str(stdout),
#             'stderr' : str(stderr),
#             'params' : str(params)}
        }
from pyrlcade_hyperopt_tests.mongodb_machine import mongodb_machine
trials = MongoTrials(mongodb_machine + 'pyrlcade_newton_rbf_tests/jobs',exp_key='pyrlcade_rbftest')

best = fmin(objective, space, trials=trials, max_evals=2000,algo=rand.suggest)

print('best: ' + str(best))
