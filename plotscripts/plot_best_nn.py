import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyrlcade.misc.save_h5py import save_results,load_results
import sys

def plot_results(results):
    res = np.array(results['r_sum_avg_list'])
    plt.plot(np.arange(res.shape[0])+1,res)

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
    plot_list = []
    plot1 = {
        'plot_files' :  ['../results/pyrlcade_gamma_testv1.1.h5py'
                      ,'../results/pyrlcade_gamma_testv1.2.h5py'
                      ,'../results/pyrlcade_gamma_testv1.3.h5py'
                      ,'../results/pyrlcade_gamma_testv1.4.h5py'
                      ,'../results/pyrlcade_gamma_testv1.5.h5py'
                      ,'../results/pyrlcade_gamma_testv1.6.h5py'
                      ,'../results/pyrlcade_gamma_testv1.7.h5py'
                      ,'../results/pyrlcade_gamma_testv1.8.h5py'
                      ,'../results/pyrlcade_gamma_testv1.9.h5py'
                      ],
        'legend' : ['gamma 0.95'
                 ,'gamma 0.97'
                 ,'gamma 0.98'
                 ,'gamma 0.99'
                 ,'gamma 0.995'
                 ,'gamma 0.997'
                 ,'gamma 0.999'
                 ,'gamma 0.9992'
                 ,'gamma 0.9995'],
        'ylabel' : 'Average Reward',
        'xlabel' : 'Episode',
        'data_name' : 'r_sum_avg_list',
        'axis' : [0,8000,-21,21],
        'save_fname' : '../result_images/pyrlcade_gamma_test_results.png'
    }

    plot2 = {
        'plot_files' :  ['../results/pyrlcade_alpha_testv1.1.h5py'
                      ,'../results/pyrlcade_alpha_testv1.2.h5py'
                      ,'../results/pyrlcade_alpha_testv1.3.h5py'
                      ,'../results/pyrlcade_alpha_testv1.4.h5py'
                      ,'../results/pyrlcade_alpha_testv1.5.h5py'
                      ,'../results/pyrlcade_alpha_testv1.6.h5py'
                      ],
        'legend' : ['alpha 0.1'
                 ,'alpha 0.2'
                 ,'alpha 0.3'
                 ,'alpha 0.4'
                 ,'alpha 0.5'
                 ,'alpha 0.6'],
        'ylabel' : 'Average Reward',
        'xlabel' : 'Episode',
        'data_name' : 'r_sum_avg_list',
        'axis' : [0,8000,-21,21],
        'save_fname' : '../result_images/pyrlcade_alpha_test_results.png'
    }
 


    plot_list.append(plot1)
    plot_list.append(plot2)

    for p in plot_list:
        for f in p['plot_files']:
            res_filename = f
            print('loading: ' + res_filename)
            res = load_results(res_filename)
            plot_results(res)
        plt.legend(p['legend'],'upper left',prop={'size': 6})
        plt.axis(p['axis'])

        plt.xlabel(p['xlabel'])
        plt.ylabel(p['ylabel'])
        plt.grid()
        plt.savefig(p['save_fname'],dpi=400,bbox_inches='tight')
        plt.clf()
