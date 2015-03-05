#this class implements a tabular storage for a Qsa table
import numpy as np

divs = np.array([33,5,1,1,5])
mins = np.array([0x32,0x26,9,7,0x26])
maxs = np.array([0xCD,0xCB,11,13,0xCB])

def pong_ram_extractor(ram):
    #ram values
    #0x31 ball x position 32-CD
    #0x36 ball y position (player's was 26-cB)
    #0x38 ball y velocity -5 to 5
    #0x3A ball x velocity -5 to 5
    #0x3c player y 26-CB
    state = np.array(ram[[0x31,0x36,0x38,0x3A,0x3C]],dtype=np.uint8)
    #these are signed bytes
    state[2] += 10
    state[3] += 10
    return state

if __name__ == '__main__':
    pass
