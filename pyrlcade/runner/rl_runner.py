#!/usr/bin/env python
import time
import math
import os
import sys
import numpy as np
from pyrlcade.env.pyrlcade_environment import pyrlcade_environment
from pyrlcade.state.tabular_ram_qsa import tabular_ram_qsa
from pyrlcade.state.nnet_qsa import nnet_qsa
from pyrlcade.state.nnet_qsa_allactions import nnet_qsa_allactions
from pyrlcade.misc.clear import clear
from pyrlcade.misc.save_h5py import save_results,load_results
from pyrlcade.state.q_learning_updater import q_learning_updater
from pyrlcade.state.sarsa_updater import sarsa_updater
from pyrlcade.state.pong_ram_extractor import pong_ram_extractor
from pyrlcade.state.normalization_transformer import normalization_transformer
from pyrlcade.state.state_expander_transformer import state_expander_transformer

class rl_runner(object):
    def run_sim(self,p):
        #init random number generator from seed
        np.random.seed(p['random_seed']);
   
        #initialize environment
        self.sim = pyrlcade_environment()
        self.sim.init(p['rom_file'],p['ale_frame_skip'])

        self.init_sim(p)

        self.do_vis = p['do_vis']
        self.save_images = p.get('save_images',False)
        self.image_save_dir = p.get('image_save_dir',None)
        save_interval = p['save_interval']

        self.showevery = p['showevery']
        self.fastforwardskip = 5

        self.reward_multiplier = p['reward_multiplier']

        #This flag is set to true if we get NaN values somewhere, which isn't supposed to happen
        self.nan_output = False

        if(self.do_vis):
            #only import if we need it, since we don't want to require installation of pygame
            from pyrlcade.vis.visualize_sdl import visualize_sdl
            v = visualize_sdl()
            v.init_vis(p)

        print_update_timer = time.time()
        start_time = time.time()
        elapsed_time = time.time()
        step_duration_timer = time.time()
        save_time = time.time()
        avg_step_duration = 1.0

        ##repeat for each episode
        self.r_sum_avg      = p['initial_r_sum_avg']
        self.r_sum_list     = []
        self.r_sum_avg_list = []

        while 1:
            self.step = 0 
            ##initialize s
            self.sim.reset_state()
            self.s = self.state_ram_extractor.extract_state(self.sim.get_state())
            #choose a from s using policy derived from Q
            (self.a,self.qsa_tmp) = self.choose_action(self.s,p);

            r_list = []
            self.r_sum = 0.0
            #repeat steps
            quit = False
            save_and_exit = False

            while 1:
                #print("s: " + str(self.s))
                ##take action a, observe r, s'
                self.sim.set_action(self.a)

                is_terminal = self.sim.step()
                #print("Terminal: " + str(self.sim.is_terminal))
                self.r = self.sim.get_reward()
                self.s_prime = self.state_ram_extractor.extract_state(self.sim.get_state())
                self.r_sum += self.r

                #choose a' from s' using policy derived from Q
                (self.a_prime,self.qsa_prime_list) = self.choose_action(self.s_prime,p)
                
                #Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s_prime,a_prime) - Q(s,a)]
                #todo: qsa_prime can be saved and reused for qsa_tmp
                #qsa_tmp = self.qsa.load(self.s,self.a)
                given_reward = self.r*self.reward_multiplier
                self.qsa_learner.update(self.alpha,self.s,self.a,given_reward,self.s_prime,self.a_prime,self.qsa_prime_list)
                #self.qsa_learner.store(self.s,self.a,self.qsa_tmp +  \
                #    self.alpha*(self.r + self.gamma*self.qsa_learner.load(self.s_prime,self.a_prime) - self.qsa_tmp))

                
                if(self.do_vis):
                    self.stats = {}
                    self.stats['action'] = self.a
                    self.stats['total_reward'] = self.r_sum
                    self.stats['episode'] = self.episode
                    self.stats['r_sum_avg'] = self.r_sum_avg
                    self.stats['learning_rate'] = self.alpha
                    self.stats['gamma'] = p['gamma']
                    self.stats['epsilon'] = self.epsilon
                    self.stats['epsilon_min'] = p['epsilon_min']
                    self.stats['save_images'] = p['save_images']
                    self.stats['image_save_dir'] = p['image_save_dir']

                    self.stats['state'] = np.copy(self.s)
                    if(p['qsa_type'] == 'nnet'):
                        self.stats['nnet_state'] = np.copy(self.qsa.net.input)
                    #show full episode for episodes that don't fast forward
                    if not (self.episode % self.showevery):
                        self.fast_forward = False
                        self.stats['fast_forward'] = False
                        if(not p['save_images']):
                            v.delay_vis()
                        v.draw_pyrlcade(self.sim.ale,self.stats)
                        exit = v.update_vis()
                        if(exit):
                            quit=True
                    #show the every nth step of each episode when fast forwarding
                    elif(is_terminal and not (self.episode % self.fastforwardskip)):
                        self.fast_forward = True
                        self.stats['fast_forward'] = True
                        if(not p['save_images']):
                            v.delay_vis()
                        v.draw_pyrlcade(self.sim.ale,self.stats)
                        exit = v.update_vis()
                        if(exit):
                            quit=True
                        
                #TODO: put this printout stuff in a function
                #the self.episode > 0 check prevents a bug where some of the printouts are empty arrays before the first episode completes
                if(print_update_timer < time.time() - 1.0 and self.episode > 0):
                    clear()
                    print("Simname: " + str(p['simname']))
                    print("Version: " + str(p['version']))
                    print("Episodes Elapsed: " + str(self.episode))
                    print("Average Reward Per Episode: " + str(self.r_sum_avg))
                    print("Epsilon: " + str(self.epsilon))
                    print("Epsilon Decay: " + str(p['epsilon_decay']))
                    print("Epsilon Min: " + str(p['epsilon_min']))
                    print("Gamma: " + str(p['gamma']))
                    print("Alpha (learning rate): " + str(self.alpha))
                    if(p.has_key('learning_rate_decay')):
                        print("Alpha (learning rate) decay: " + str(p['learning_rate_decay']))

                    if(p.has_key('learning_rate_min')):
                        print("Alpha (learning rate) Min: " + str(p['learning_rate_min']))
                    if(p.has_key('activation_function')):
                        print("activation_function: " + str(p['activation_function']))
                    if(p.has_key('activation_function_final')):
                        print("activation_function_final: " + str(p['activation_function_final']))
                    if(p.has_key('num_hidden')):
                        print("num_hidden: " + str(p['num_hidden']))
                    if(p.has_key('rbf_transform_size')):
                        print("rbf_transform_size: " + str(p['rbf_transform_size']))
                    if(p.has_key('rbf_width_scale')):
                        print("rbf_width_scale: " + str(p['rbf_width_scale']))
                    if(p.has_key('cluster_func') and p['cluster_func'] is not None):
                        print("clusters_selected: " + str(p['clusters_selected']))
                    print("Reward Multiplier: " + str(p['reward_multiplier']))
                    print("RL Algorithm: " + str(p['rl_algo']))


                    print("Average Steps Per Second: " + str(1.0/avg_step_duration))
                    print("Action Type: " + str(p['action_type']))
                    print("a_list: " + str(self.tmp_a_list))
                    m, s = divmod(time.time() - start_time, 60)
                    h, m = divmod(m, 60)
                    print "Elapsed Time %d:%02d:%02d" % (h, m, s)
                    sys.stdout.flush()
                    print_update_timer = time.time()


                if(self.episode >= p['train_episodes']):
                    save_and_exit = True
                    quit=True

                if(quit):
                    break
                if(is_terminal):
                    break
                if(self.step > p['max_steps']):
                    break
                #exit if the neural network starts outputting NaN
                if(np.isnan(np.sum(self.qsa_prime_list))):
                    print("ERROR: The qsa_prime has NaN values! Something went unstable!")
                    self.nan_output=True
                    save_and_exit=True
                    break
                ## s <- s';  a <-- a'
                self.s = self.s_prime
                self.a = self.a_prime
                self.qsa_tmp = self.qsa_prime_list

                #print("Next Step \n")
                self.step += 1
                avg_step_duration = 0.995*avg_step_duration + (1.0 - 0.995)*(time.time() - step_duration_timer)
                step_duration_timer = time.time()
                #end step loop

            if(p['decay_type'] == 'geometric'):
                self.epsilon = self.epsilon * p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)
            elif(p['decay_type'] == 'linear'):
                self.epsilon = self.epsilon - p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)

            if(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'geometric'):
                self.alpha = self.alpha * p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min'],self.alpha)
            elif(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'linear'):
                self.alpha = self.alpha - p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min'],self.alpha)


            if(time.time() - save_time > save_interval or save_and_exit == True):
                print('saving results...')
                self.save_results(p['results_dir'] + p['simname'] + p['version'] + '.h5py',p)
                save_time = time.time();

            if(quit==True or save_and_exit==True):
                break;

            #compute the number of steps that have a positive reward, as the number of steps that balanced
            self.r_sum_avg = 0.995*self.r_sum_avg + (1.0 - 0.995)*self.r_sum

            self.r_sum_list.append(self.r_sum) 
            self.r_sum_avg_list.append(self.r_sum_avg) 
            self.episode += 1
            #end episode loop

        self.update_results(p)
        #print("obj: " + str(obj) + " argmax: " + str(argmax))
        return self.results

    def choose_action(self,state,p):
        max_action = -1e99
        
        #epsilon-greedy
        if(p['action_type'] == 'e_greedy'):
            qsa_list = self.qsa_learner.get_qsa_list(state)
            if(np.random.random() < self.epsilon):
                a = np.random.randint(self.num_actions)
                #print("selected action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
            else:
                a = np.argmax(np.array(qsa_list))
                #print("selected random action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
        elif(p['action_type'] == 'noisy_qsa'):
            #INIT CODE HERE
            if(self.step == 0 and self.episode == 0):
                self.qsa_std_avg = p['qsa_avg_init']
                self.qsa_avg_alpha = p['qsa_avg_alpha']
                #this will give a moving average estimate of the probability of selecting a different action
                #(used for printing only)
                self.prob_of_different_action = 0.0
            qsa_list = np.array([self.qsa.load(state,i) for i in range(self.num_actions)])
            qsa_std = np.std(qsa_list)
            self.qsa_std_avg = self.qsa_avg_alpha*self.qsa_std_avg + (1.0 - self.qsa_avg_alpha)*qsa_std
            noise = self.epsilon*self.qsa_std_avg*np.random.rand(self.num_actions)
            a_before = np.argmax(np.array(qsa_list))
            a = np.argmax(np.array(qsa_list + noise))
            self.prob_of_different_action = 0.999*self.prob_of_different_action + (1.0 - 0.999)*(a != a_before)

        #save this for printout, and to check for nan
        self.tmp_a_list = np.copy(np.array(qsa_list))
        return (a,qsa_list)

    #this updates the internal self.results variable to reflect the latests results to be either saved or returned
    def update_results(self,p):
        self.results = {}
        #TODO: save neural network weights
        if(p['qsa_type'] == 'tabular'):
            self.results['qsa_values'] = np.array(self.qsa.data)
            self.results['state_size'] = np.array(self.qsa.size)
        self.results['r_sum_list'] = np.array(self.r_sum_list)
        self.results['r_sum_avg_list'] = np.array(self.r_sum_avg_list)
        self.results['num_actions'] = np.array(self.num_actions)
        self.results['epsilon'] = np.array(self.epsilon)
        self.results['epsilon_decay'] = np.array(self.epsilon_decay)
        self.results['epsilon_min'] = np.array(self.epsilon_min)
        self.results['alpha'] = np.array(self.alpha)
        self.results['gamma'] = np.array(self.gamma)
        self.results['nan_output'] = np.array(self.nan_output)
        self.results['episode'] = np.array(self.episode)
        self.results['os'] = os.uname()[1]
        self.results['parameters'] = p
        #TODO: save and load more hyperparameters, such as game memory?

    def save_results(self,filename,p):
        self.update_results(p)
        #skip saving if the parameter says not to save
        if(p.has_key('skip_saving') and p['skip_saving'] == True):
            return
        print("saving: " + str(filename))
        save_results(filename,self.results)

#TODO: THIS FUNCTION IS BROKEN! LOADING MAY NOT WORK PROPERLY!
#      rework this, to support neural network architecture
#    def load_results(self,filename,p):
#        self.results = load_h5py(filename,p)

#        self.epsilon = self.results['epsilon'].value
#        self.epsilon_decay = self.results['epsilon_decay'].value
#        self.epsilon_min = self.results['epsilon_min'].value
#        self.alpha = self.results['alpha'].value
#        self.gamma = self.results['gamma'].value
#        self.state_min = list(self.results['state_min'])
#        self.state_max = list(self.results['state_max'])
#        self.state_size = list(self.results['state_size'])
#        self.episode = self.results['episode'].value
#        self.num_actions = 3
#        self.qsa = tabular_qsa()
#        self.qsa.init(self.state_min,self.state_max,self.state_size,self.num_actions)
#        self.qsa.data = np.array(self.results['qsa_values'])
#        print('loaded epsilon: ' + str(self.epsilon))

    def init_sim(self,p):
        self.epsilon = p['epsilon']
        self.epsilon_decay = p.get('epsilon_decay',1.0)
        self.epsilon_min = p.get('epsilon_min',self.epsilon)
        self.gamma = p['gamma']

        self.episode = 0

        self.num_actions = self.sim.ale.getMinimalActionSet().size

        self.state_ram_extractor = pong_ram_extractor()

        (state_size,state_mins,state_maxs) = self.state_ram_extractor.get_size_and_range()

        self.alpha = p['learning_rate']
        if(p.has_key('nnet_use_combined_actions') and p['nnet_use_combined_actions'] == True):
            self.use_combined_actions = True
        else:
            self.use_combined_actions = False

        if(p['qsa_type'] == 'tabular'):
            self.qsa = tabular_ram_qsa()
            self.qsa.init(state_mins,state_maxs,self.num_actions,p)
        elif(p['qsa_type'] == 'nnet'):
            if(p.has_key('nnet_use_combined_actions') and p['nnet_use_combined_actions'] == True):
                self.qsa = nnet_qsa_allactions()
            else:
                self.qsa = nnet_qsa()
            mins = np.ones(state_size)*(-1.25)
            maxs = np.ones(state_size)*(1.25)

            #Neural network input must be normalized, set up a normalizer for the state extractor
            self.normalization_transformer = normalization_transformer()
            self.normalization_transformer.init(state_mins,state_maxs,mins,maxs)
            self.state_ram_extractor.set_transform_class(self.normalization_transformer)

            nnet_size = state_size
            #set up a radial basis network transformer
            if(p.has_key('do_rbf_transform') and p['do_rbf_transform'] == True):
                self.rbf_transformer = state_expander_transformer()
                self.rbf_size = p['rbf_transform_size']
                self.rbf_width_scale = p['rbf_width_scale']
                self.rbf_transformer.init(mins,maxs,self.rbf_size,self.rbf_width_scale)
                self.normalization_transformer.set_transform_class(self.rbf_transformer)
                nnet_size = self.rbf_size
            self.qsa.init(nnet_size,self.num_actions,p)

        if(p['rl_algo'] == 'sarsa'):
            self.qsa_learner = sarsa_updater()
        else:
            self.qsa_learner = q_learning_updater()
        self.qsa_learner.init(self.qsa,self.gamma,self.use_combined_actions) 


if __name__ == '__main__':
    g = rl_runner()
    p = {}
    g.run_sim(p)
