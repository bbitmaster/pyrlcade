#!/usr/bin/env python
import pygame
import math
import numpy as np
import matplotlib.cm as cm
import pyrlcade
from ale_python_interface import ALEInterface
from pyrlcade.state.pong_ram_extractor import pong_ram_extractor

class visualize_pong_qsa_sdl(object):
    def init_vis(self,p):
        pygame.init()
        (display_width,display_height) = (1280,720)
        self.screen = pygame.display.set_mode((display_width,display_height))
        pygame.display.set_caption("Arcade Learning Environment Agent Display")
        self.game_surface = None

        self.clock = pygame.time.Clock()
        self.delay = p['fps']
        self.framenum = 0
        self.i_range = [-0.001,0.001]
        #precompute pallete
        self.intensity_pal = cm.afmhot(np.arange(256))
        self.intensity_pal = (self.intensity_pal*255).astype(np.int)
    #call this every iteration to slow down to real time
    def delay_vis(self):
        if(self.delay >= 0):
            self.clock.tick(self.delay)

    #call this to update the visualization event loop
    #return True if the user wants to exit. return False otherwise
    def update_vis(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True;
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True;
        return False

    def get_keys(self):
        keys = [];
        pressed = pygame.key.get_pressed()
        keys.append(pressed[pygame.K_z])
        keys.append(pressed[pygame.K_LEFT])
        keys.append(pressed[pygame.K_RIGHT])
        keys.append(pressed[pygame.K_UP])
        keys.append(pressed[pygame.K_DOWN])
        return keys

    #call this before calling update vis
    def draw_pyrlcade(self,ale,stats):
        #Init game surface here to avoid needing game screen dims in the init function
        if(self.game_surface is None):
            (screen_width,screen_height) = ale.getScreenDims()
            self.game_surface = pygame.Surface((screen_width,screen_height))

        #clear screen
        self.screen.fill((0,0,0))

        #get atari screen pixels and blit them
        numpy_surface = np.frombuffer(self.game_surface.get_buffer(),dtype=np.int32)
        ale.getScreenRGB(numpy_surface)
        del numpy_surface
        self.screen.blit(pygame.transform.scale(self.game_surface,(self.game_surface.get_width()*3,self.game_surface.get_height()*3)),(0,0))

        #get RAM
        ram_size = ale.getRAMSize()
        ram = np.zeros((ram_size),dtype=np.uint8)
        ale.getRAM(ram)
 
        #display QSA
        line_pos = 0
        font = pygame.font.SysFont("Ubuntu Mono",30)
        text = font.render("QSA Sliced Grid View of 5 dimensional function" ,1,(208,255,255))
        height = font.get_height()*1.2
        self.screen.blit(text,(490,line_pos))
        line_pos += height

        font = pygame.font.SysFont("Ubuntu Mono",20)
        text = font.render("Intensity range: %2.4f , %2.4f" % (self.i_range[0],self.i_range[1]),1,(255,255,255))
        height = font.get_height()*1.2
        self.screen.blit(text,(490,line_pos))
        line_pos += height

        #Get the pong state variables from ram
        #display qsa for one slice
        qsa_learner = stats['qsa_learner']
        state_ram_extractor = stats['state_ram_extractor']
        base_loc = (490,line_pos)

        #1. Ball moving (toward player, toward opponnent)
        for ball_horz_vel in (1,2,4,5):
            #   1.ball moving vertically (down,no_motion,up)
            for ball_vert_vel in (0,1,2):
            #       1.on (far left, mid-left, mid-right, right) side
                for ball_x_pos in (0,1,2,3,4):
                    base_state = np.array([ball_x_pos,0,ball_vert_vel,ball_horz_vel,0])
                    base_state = base_state.transpose()
                    horz_index = [0,0,1,0,2,3][ball_horz_vel]
                    self.show_qsa_slice((base_loc[0] + (horz_index)*34 + (ball_x_pos)*145,base_loc[1] + ball_vert_vel*34),base_state,qsa_learner,state_ram_extractor)

        #display qsa pallete
        for x in range(256):
            for y in range(20):
                loc = (490+x,int(line_pos+110+y))
                self.screen.set_at(loc,self.intensity_pal[x,0:3])

        line_pos += 140
        
        #display current action
        if(stats is not None):
            font = pygame.font.SysFont("Ubuntu Mono",32)
            text = font.render("Current Action: " + str(stats['action']) ,1,(208,208,255))
            height = font.get_height()*1.2
            self.screen.blit(text,(490,line_pos))
            line_pos += height

            #display reward
            font = pygame.font.SysFont("Ubuntu Mono",30)
            text = font.render("Total Reward: " + str(stats['total_reward']) ,1,(208,255,255))
            self.screen.blit(text,(490,line_pos))
            line_pos += height

            #display state
            font = pygame.font.SysFont("Ubuntu Mono",20)
            state = stats['state'][0:20]
            if(stats['state'].dtype == np.int64):
                text_str = "State: " + ''.join(["%02d "%x for x in state])
            else:
                text_str = "State: " + ''.join(["%8.4f "%x for x in state])
            text = font.render(text_str,1,(208,208,255))
            self.screen.blit(text,(490,line_pos))

            #display episodes below game
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Episode: " + str(stats['episode']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #alpha
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Average Reward: " + str(stats['r_sum_avg']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #alpha
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Learning Rate: " + str(stats['learning_rate']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #gamma
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Gamma: " + str(stats['gamma']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #current epsilon
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Epsilon: " + str(stats['epsilon']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #min epsilon
            line_pos += height
            font = pygame.font.SysFont("Ubuntu Mono",25)
            text = font.render("Epsilon Minimum: " + str(stats['epsilon_min']) ,1,(255,255,255))
            self.screen.blit(text,(490,line_pos))

            #display nnet state
            if(stats.has_key('nnet_state')):
                font = pygame.font.SysFont("Ubuntu Mono",20)
                line_pos=640
                state_pos=0
                state = stats['nnet_state']
                state_size = state.size
                join_str = "NNet State "
                while(state_pos < state_size):
                    text_str = join_str + ''.join(["%3.3f "%state[x] for x in range(state_pos,min(state_pos+18,state_size))])
                    text = font.render(text_str,1,(208,208,255))
                    if(line_pos > self.game_surface.get_height):
                        break
                    self.screen.blit(text,(18,line_pos))
                    join_str = ''
                    line_pos += font.get_height()
                    state_pos += 18

            #display bias weights (if applicable)
            #if(type(stats['qsa']) is pyrlcade.state.nnet_qsa.nnet_qsa):
            #    qsa = stats['qsa']
            #    print(qsa.net.layer[1].weights[:,-1])
            if(stats['save_images']):
                pygame.image.save(self.screen,stats['image_save_dir'] + "frame_" + str(self.framenum) + ".png")
                self.framenum = self.framenum + 1           

        pygame.display.flip()

    def show_qsa_slice(self,base_loc,state,qsa_learner,state_ram_extractor):
        #draw plot for state variables
        new_range = [self.i_range[0],self.i_range[1]]
        for x in range(33):
            for y in range(33):
                loc = (int(base_loc[0] + x),int(base_loc[1] + y))
                state[1] = x
                state[4] = y
                if(state_ram_extractor.transform_class is not None): 
                    base_state = state_ram_extractor.transform_class.transform(state)
                    #print("base_state: " + str(base_state))
                    #print("state     : " + str(base_state))
                else:
                    base_state = state
                qsa_list = qsa_learner.get_qsa_list(base_state)
                intensity = np.max(qsa_list)
                new_range[0] = min(self.i_range[0],intensity)
                new_range[1] = max(self.i_range[1],intensity)

                intensity = (intensity - self.i_range[0])/(self.i_range[1] - self.i_range[0])
                #should we square intensity here? for Gamma correction
                #intensity = intensity**2
                intensity = int(intensity*256)
                intensity = max(intensity,0)
                intensity = min(intensity,255)
                self.screen.set_at(loc,self.intensity_pal[intensity,0:3])
        self.i_range = new_range




#this runs a simple keyboard driven test, with no simulator for the cart-pole
if __name__ == '__main__':
    import sys
    key_action_tform_table = (
    0, #00000 none
    2, #00001 up
    5, #00010 down
    2, #00011 up/down (invalid)
    4, #00100 left
    7, #00101 up/left
    9, #00110 down/left
    7, #00111 up/down/left (invalid)
    3, #01000 right
    6, #01001 up/right
    8, #01010 down/right
    6, #01011 up/down/right (invalid)
    3, #01100 left/right (invalid)
    6, #01101 left/right/up (invalid)
    8, #01110 left/right/down (invalid)
    6, #01111 up/down/left/right (invalid)
    1, #10000 fire
    10, #10001 fire up
    13, #10010 fire down
    10, #10011 fire up/down (invalid)
    12, #10100 fire left
    15, #10101 fire up/left
    17, #10110 fire down/left
    15, #10111 fire up/down/left (invalid)
    11, #11000 fire right
    14, #11001 fire up/right
    16, #11010 fire down/right
    14, #11011 fire up/down/right (invalid)
    11, #11100 fire left/right (invalid)
    14, #11101 fire left/right/up (invalid)
    16, #11110 fire left/right/down (invalid)
    14  #11111 fire up/down/left/right (invalid)
    )
    ale = ALEInterface()

    v = visualize_sdl()

    p = {}
    p['fps'] = 60
    v.init_vis(p)
    ale.loadROM(sys.argv[1])
    while 1:
        v.delay_vis()
        pressed = v.get_keys()
        keys = 0
        keys |= pressed[3]
        keys |= pressed[4]  << 1
        keys |= pressed[1]  << 2
        keys |= pressed[2]  << 3
        keys |= pressed[0]  << 4
        a = key_action_tform_table[keys]

        reward = ale.act(a);

        v.draw_pyrlcade(ale,None)
        exit = v.update_vis()
        if(ale.game_over()):
            exit = True
        if(exit):
            break

