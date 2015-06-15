#!/usr/bin/env python
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyrlcade.misc.save_h5py import save_results,load_results
import sys

def plot_results(results,color=None):
    res = np.array(results['r_sum_avg_list'])
    #if color is not None:
    #    plt.plot(np.arange(res.shape[0])+1,res)
    #else:
    plt.plot(np.arange(res.shape[0])+1,res,color=color)

def calc_polyfit(results):
    steps = np.array(results['r_sum_avg_list'])
    x = np.arange(steps.shape[0])+1
    z = np.polyfit(x,steps,6)
    poly_z = np.poly1d(z)
    err = (steps - poly_z(x))
    error_std = np.sum(err**2)/20000.
    print("error_std: " + str(error_std))
    xp = np.linspace(1,20000,500)
    plt.plot(x,poly_z(x))
    return 'polyfit mse: ' + str(error_std)
    

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        params_file = sys.argv[1]
    else:
        print("Running with default parameter file: pyrlcade_default_params.py")
        sys.exit()

    p = {}
    execfile(params_file,p)
    
    for plot in p['plot_list']:
        for i,f in enumerate(plot['plot_files']):
            res_filename = f
            print('loading: ' + res_filename)
            res = load_results(res_filename)
            #if(plot.has_key('use_hsv') and plot['use_hsv'] == True):
            idx = float(i)/float(len(plot['plot_files']))
            print("i: " + str(i) + " idx: " + str(idx))
            plot_results(res,color=plt.cm.hsv(idx))
            #else:
            #print("blah")
            #plot_results(res)
        plt.legend(plot['legend'],plot.get('location','upper left'),prop={'size': 6})
        plt.axis(plot['axis'])
        if(plot.has_key('title')):
            plt.title(plot['title'])

        plt.xlabel(plot['xlabel'])
        plt.ylabel(plot['ylabel'])
        plt.grid()
        plt.savefig(plot['save_fname'],dpi=400,bbox_inches='tight')
        plt.clf()
