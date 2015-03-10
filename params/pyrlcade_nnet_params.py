from math import exp, log
#cartpole_default_params
runtype='sarsa'

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'pyrlcade_nnet_test'
version = '1.1'
results_dir = '../results/'

rom_file='/home/bgoodric/research/python/pyrlcade/roms/pong.bin'

#if load_name is set, then the simulation will load this file and resume from there, this is useful for watching the behavior of a trained agent
#load_name = '../results/cartpole_sarsa_test1.1.h5py'

data_dir = '../data/'
save_interval = 15*60

#run for a total number of episodes
train_episodes=50000
max_steps=10000

use_float32=True

random_seed = 4;
initial_r_sum_avg=-21.0

save_images=False
image_save_dir="/home/bgoodric/tmp/" #I Guess that underutilized windows partitition with all that storage is good for something...

qsa_type='nnet'

#parameters for neural network qsa
activation_function='linear_rectifier'

num_hidden=64
learning_rate = 0.01
learning_rate_decay_type='geometric'
learning_rate_decay=1.0
learning_rate_min=0.0001
momentum=0.0
maxnorm=None
dropout=None

initialization_scheme='glorot'
initialization_scheme_final='glorot'

initialization_constant=1.0
initialization_constant_final=1.0

#action is encoded using one hot encoding with these as the "hot" and "not hot" targets
incorrect_target = -1.0
correct_target = 1.0

#decay_type can be 'geometric' or 'linear'
decay_type='geometric'
epsilon=0.20
epsilon_min=0.01
#epsilon_decay=exp((log(epsilon_min) - log(epsilon))/10000.0)
#print("epsilon_decay: " + str(epsilon_decay))
epsilon_decay=0.9998
#epsilon_decay = (epsilon - epsilon_min)/10000
gamma=0.999

action_type='e_greedy'

#If defined, will print the state variables on every frame
print_state_debug=True

#in sarsa mode, this tells if the SDL display should be enabled. Set to False if the machine does not have pygame installed
do_vis=False

#in sarsa mode, this tells how often to display, -1 for none
showevery=500

#these affect the display. They tell the size in pixels of the display, the axis size, and how many frames to skip
display_width=1280
display_height=720
axis_x_min=-10.0
axis_x_max=10.0
axis_y_min=-5.5
axis_y_max=5.5
fps=60
ale_frame_skip=4
