#!/usr/bin/env python
import time
import math
import sys
import numpy as np
from pyrlcade.env.pyrlcade_environment import pyrlcade_environment
from pyrlcade.state.tabular_ram_qsa import tabular_ram_qsa
from pyrlcade.misc.clear import clear
from pyrlcade.misc.save_h5py import save_results,load_results
from pyrlcade.state.q_learning_updater import q_learning_updater
import pyrlcade.state.pong_ram_extractor as pong_ram_extractor

class rl_runner(object):
    def run_sim(self,p):
        #init random number generator from seed
        np.random.seed(p['random_seed']);
   
        #initialize environment
        self.sim = pyrlcade_environment()
        self.sim.init(p['rom_file'],p['ale_frame_skip'])

        #initialize hyperparameters fresh, unless we are resuming a saved simulation
        #in which case, we load the parameters
        if(not p.has_key('load_name')):
            self.init_sim(p)
        else:
            self.load_sim(p)

        self.do_vis = p['do_vis']
        self.save_images = p.get('save_images',False)
        self.image_save_dir = p.get('image_save_dir',None)
        save_interval = p['save_interval']

        self.showevery = p['showevery']
        self.fastforwardskip = 5

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
        self.r_sum_avg = -0.95
        self.r_sum_avg_list = []

        while 1:
            self.step = 0 
            ##initialize s
            self.sim.reset_state()
            self.s = self.ram_extractor_func(self.sim.get_state())
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
                self.s_prime = self.ram_extractor_func(self.sim.get_state())
                self.r_sum += self.r
                r_list.append(self.r)

                #choose a' from s' using policy derived from Q
                (self.a_prime,self.qsa_prime_list) = self.choose_action(self.s_prime,p)
                
                #Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s_prime,a_prime) - Q(s,a)]
                #todo: qsa_prime can be saved and reused for qsa_tmp
                #qsa_tmp = self.qsa.load(self.s,self.a)
                self.qsa_learner.update(self.s,self.a,self.r,self.s_prime,self.a_prime,self.qsa_prime_list)
                #self.qsa_learner.store(self.s,self.a,self.qsa_tmp +  \
                #    self.alpha*(self.r + self.gamma*self.qsa_learner.load(self.s_prime,self.a_prime) - self.qsa_tmp))

                
                if(self.do_vis):
                    self.stats = {}
                    self.stats['action'] = self.a
                    self.stats['total_reward'] = self.r_sum
                    disp_state = np.copy(self.s)
                    disp_state = disp_state - self.qsa.mins
                    disp_state /= self.qsa.divs
                    disp_state = np.minimum(disp_state,self.qsa.arr_maxs)
                    disp_state = np.maximum(disp_state,self.qsa.arr_mins)

                    self.stats['state'] = np.copy(disp_state)
                    #show full episode for episodes that don't fast forward
                    if not (self.episode % self.showevery):
                        self.fast_forward = False
                        v.delay_vis()
                        v.draw_pyrlcade(self.sim.ale,self.stats)
                        exit = v.update_vis()
                        if(exit):
                            quit=True
                    #show the every nth step of each episode when fast forwarding
                    elif(self.step == 0 and not (self.episode % self.fastforwardskip)):
                        self.fast_forward = True
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
                    print("Episodes Elapsed: " + str(self.episode))
                    print("Average Reward Per Episode: " + str(self.r_sum_avg))
                    print("Epsilon: " + str(self.epsilon))
                    print("Epsilon Min: " + str(p['epsilon_min']))
                    print("Alpha (learning rate): " + str(self.alpha*p['learning_rate']))
                    if(p.has_key('learning_rate_decay')):
                        print("Alpha (learning rate) decay: " + str(p['learning_rate_decay']))
                    if(p['action_type'] == 'noisy_qsa'):
                        print("Average QSA Standard Deviation: " + str(self.qsa_std_avg))
                        print("Probability of taking different action: " + str(self.prob_of_different_action))
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
                ## s <- s';  a <-- a'
                self.s = self.s_prime
                self.a = self.a_prime
                #self.qsa_tmp = self.qsa_prime_list

                #print("Next Step \n")
                self.step += 1
                avg_step_duration = 0.995*avg_step_duration + (1.0 - 0.995)*(time.time() - step_duration_timer)
                step_duration_timer = time.time()
                #end step loop

            #compute the number of steps that have a positive reward, as the number of steps that balanced
            self.r_sum_avg = 0.995*self.r_sum_avg + (1.0 - 0.995)*self.r_sum
            
            if(p['decay_type'] == 'geometric'):
                self.epsilon = self.epsilon * p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)
            elif(p['decay_type'] == 'linear'):
                self.epsilon = self.epsilon - p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)
            

            if(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'geometric'):
                self.alpha = self.alpha * p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min']/p['learning_rate'],self.alpha)
            elif(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'linear'):
                self.alpha = self.alpha - p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min']/p['learning_rate'],self.alpha)


            #save stuff (TODO: Put this in a save function)
            if(time.time() - save_time > save_interval or save_and_exit == True):
                print('saving results...')
                self.save_results(p['results_dir'] + p['simname'] + p['version'] + '.h5py',p)
                save_time = time.time();

            if(quit==True or save_and_exit==True):
                break;
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

        self.tmp_a_list = np.copy(np.array(qsa_list))
        return (a,qsa_list)

    #this updates the internal self.results variable to reflect the latests results to be either saved or returned
    def update_results(self,p):
        self.results = {}
        #TODO: save neural network weights
        if(p['qsa_type'] == 'tabular'):
            self.results['qsa_values'] = np.array(self.qsa.data);
            self.results['state_size'] = np.array(self.qsa.size);
        self.results['num_actions'] = np.array(self.num_actions);
        self.results['epsilon'] = np.array(self.epsilon)
        self.results['epsilon_decay'] = np.array(self.epsilon_decay)
        self.results['epsilon_min'] = np.array(self.epsilon_min)
        self.results['alpha'] = np.array(self.alpha)
        self.results['gamma'] = np.array(self.gamma)
        self.results['episode'] = np.array(self.episode)
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
        #TODO: this should depend on hyperparameters, and should be different for nnet (for normalization)
        self.state_ram_extractor = pong_ram_extractor
        self.ram_extractor_func = pong_ram_extractor.pong_ram_extractor

        if(p['qsa_type'] == 'tabular'):
            self.qsa = tabular_ram_qsa()
            divs = self.state_ram_extractor.divs
            mins = self.state_ram_extractor.mins
            maxs = self.state_ram_extractor.maxs

            self.alpha = p['learning_rate']
            self.qsa.init(mins,maxs,divs,self.num_actions,self.alpha)
        #elif(p['qsa_type'] == 'nnet'):
        #    self.qsa = nnet_qsa()
        #    self.qsa.init(self.state_min,self.state_max,self.num_actions,p)
        #   #The neural network has its own internal learning rate (alpha is ignored)
        #    self.alpha = 1.0

        self.qsa_learner = q_learning_updater()
        self.qsa_learner.init(self.qsa,self.gamma) 


if __name__ == '__main__':
    g = rl_runner()
    p = {}
    g.run_sim(p)
