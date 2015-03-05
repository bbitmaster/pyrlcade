#!/usr/bin/env python
import numpy as np
import sys
from ale_python_interface import ALEInterface

class pyrlcade_environment(object):
    def init(self,rom_file):

        self.ale = ALEInterface()

        self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
        self.ale.set("random_seed",123)

        self.ale.loadROM(rom_file)
        self.legal_actions = self.ale.getMinimalActionSet()
        ram_size = self.ale.getRAMSize()
        self.ram = np.zeros((ram_size),dtype=np.uint8)
        self.ale.getRAM(self.ram)

        self.state = self.ale.getRAM(self.ram)

    def reset_state(self):
        self.ale.reset_game()

    def set_action(self,a):
        self.action = a

    def step(self):
        self.reward = self.ale.act(self.action)
        is_terminal = self.ale.game_over()
        return is_terminal

    def get_state(self):
        self.ale.getRAM(self.ram)
        return self.ram

    def get_reward(self):
        return self.reward

