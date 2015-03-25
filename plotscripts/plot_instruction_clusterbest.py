plot_list = []
plot1 = {
    'plot_files' :  ['../results/clustertest1/pyrlcade_hyperopt_nn_clustertest_SKXDZYNBDAAM.h5py'
                    ,'../results/clustertest1/pyrlcade_hyperopt_nn_clustertest_RNDYZJRHCEZJ.h5py'
                    ,'../results/clustertest1/pyrlcade_hyperopt_nn_clustertest_BAFVILOWUMEM.h5py'
                    ,'../results/clustertest1/pyrlcade_hyperopt_nn_clustertest_GTZYMDSWFWVN.h5py'
                    ,'../results/clustertest1/pyrlcade_hyperopt_nn_clustertest_QGSQJCXHZPYL.h5py'
                    ],
    'legend' : ['Cluster-Select Best #1'
                ,'Cluster-Select Best #2'
                ,'Cluster-Select Best #3'
                ,'Cluster-Select Best #4'
                ,'Cluster-Select Best #5'
                ],
    'ylabel' : 'Average Reward',
    'xlabel' : 'Episode',
    'data_name' : 'r_sum_avg_list',
    'axis' : [0,10000,-21,21],
    'save_fname' : '../result_images/pyrlcade_clustertest_test_results.png'
}

plot2 = {
    'plot_files' :  ['../results/nnettest/pyrlcade_hyperopt_nn_nnettest_XCNBOYVJLFZS.h5py'
                    ,'../results/nnettest/pyrlcade_hyperopt_nn_nnettest_AFJRPDCGBQXY.h5py'
                    ,'../results/nnettest/pyrlcade_hyperopt_nn_nnettest_GDCMSJUKJDXH.h5py'
                    ,'../results/nnettest/pyrlcade_hyperopt_nn_nnettest_PLEWOXVNWZYY.h5py'
                    ,'../results/nnettest/pyrlcade_hyperopt_nn_nnettest_DKENIKSALUEP.h5py'
                    ],

    'legend' : ['Neural Network Best #1'
                ,'Neural Network Best #2'
                ,'Neural Network Best #3'
                ,'Neural Network Best #4'
                ,'Neural Network Best #5'
                ],

    'ylabel' : 'Average Reward',
    'xlabel' : 'Episode',
    'data_name' : 'r_sum_avg_list',
    'axis' : [0,10000,-21,21],
    'save_fname' : '../result_images/pyrlcade_nnet_test_results.png'
}
 


plot_list.append(plot1)
plot_list.append(plot2)


