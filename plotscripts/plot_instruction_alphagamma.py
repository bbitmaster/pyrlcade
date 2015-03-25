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


