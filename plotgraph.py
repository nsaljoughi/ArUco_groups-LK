import matplotlib.pyplot as plt
import numpy as np
import os

def get_num(line, startidx, sep1, sep2):
    idx1 = line[startidx:].find(sep1)
    idx2 = line[startidx + idx1 + 1:].find(sep2)
    return line[startidx + idx1 + 1:startidx + idx1 + 1 + idx2]

current_dir = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.abspath(os.path.join(current_dir, 'build'))

DIFF_FILT = []
DIFF_UNFILT = []
ROT_FILT = []
ROT_UNFILT = []

for filename in os.listdir(build_dir):
    if (filename == "results_filt.txt"):
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

        DIFF_FILT = DIFF
        ROT_FILT = DIFF_ROT

    elif (filename == "results_unfilt.txt"):
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

        DIFF_UNFILT = DIFF
        ROT_UNFILT = DIFF_ROT
        
    else:
        continue

#plt.style.use('dark_background')

fig = plt.figure('Results', figsize=(20, 14))

plt.subplot(2, 1, 1)
plt.plot(DIFF_UNFILT, color='orange', linewidth=3, label='Unfiltered')
plt.plot(DIFF_FILT, color='blue', linewidth=1.5, label='Filtered')
plt.title('Difference in translation', fontsize=15)
plt.xlabel('frames', fontsize=15)
plt.ylabel('||x(t) - x(t-1)||', fontsize=15)
plt.ylim(-0.5, 3)
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ROT_UNFILT, color='green', linewidth=3, label='Unfiltered')
plt.plot(ROT_FILT, color='red', linewidth=1.5, label='Filtered')
plt.title('Difference in rotation', fontsize=15)
plt.xlabel('frames', fontsize=15)
plt.ylabel('||q(t) - q(t-1)||', fontsize=15)
plt.grid(True)
plt.legend()

fig.tight_layout()
plt.savefig('results_plot.png', dpi=300)
plt.show()

