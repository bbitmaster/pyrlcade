plot_list = []
plot1 = {
    'plot_files' :  [
                     '../results/tabularbestvisrun1.1.h5py'
                     ,'../results/nnbestvisrun_zerobias1.1.h5py'
                     ,'../results/nnbestvisrun1.1.h5py'
                     ,'../results/rbfbestvisrun1.1.h5py'
                     ,'../results/clusternnbestvisrun1.1.h5py'
                    ],
    'legend' : ['Tabular Run'
                ,'Neural Network Run #1'
                ,'Neural Network Run #2'
                ,'Radial Basis Network Run'
                ,'Cluster-Select Run'
                ],
    'ylabel' : 'Average Reward',
    'xlabel' : 'Episode',
    'data_name' : 'r_sum_avg_list',
    'axis' : [0,10000,-21,21],
    'save_fname' : '../result_images/pyrlcade_video_results.png'
}

plot_list.append(plot1)


