import matplotlib.pyplot as plt
import numpy as np
import os

def get_num(line, startidx, sep1, sep2):
    idx1 = line[startidx:].find(sep1)
    idx2 = line[startidx + idx1 + 1:].find(sep2)
    return line[startidx + idx1 + 1:startidx + idx1 + 1 + idx2]

current_dir = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.abspath(os.path.join(current_dir, 'build'))

for filename in os.listdir(build_dir):
    if (filename == "results_filt.txt") or (filename == "results_unfilt.txt"):
        flog = os.path.abspath(os.path.join(build_dir, filename))
        with open(flog) as f:
            lines = f.readlines()

        FRAME = []
        DIFF = []
        DIFF_ROT = []

        for line in lines:
            frameidx = line.find('Frame')
            diffrotidx = line.find('Rot')
            diffidx = line.find('Dist')

            frame = get_num(line, frameidx, ' ', ',')
            diff_rot = get_num(line, diffrotidx, ' ', ',')
            diff = get_num(line, diffidx, ' ', ';')

            FRAME.append(int(frame))
            DIFF.append(float(diff))
            DIFF_ROT.append(float(diff_rot))
            
        fig = plt.figure('Results for {}'.format(filename), figsize=(9, 14))
        
        plt.subplot(2, 1, 1)
        plt.plot(DIFF, color='blue', linewidth=2)
        plt.title('Difference in translation', fontsize=10)
        plt.xlabel('frames', fontsize=10)
        plt.ylabel('||x(t) - x(t-1)||', fontsize=10)
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(DIFF_ROT, color='green', linewidth=2)
        plt.title('Difference in rotation', fontsize=10)
        plt.xlabel('frames', fontsize=10)
        plt.ylabel('||q(t) - q(t-1)||', fontsize=10)
        plt.grid(True)
        
        fig.tight_layout()
        plt.show()

    else:
        continue
