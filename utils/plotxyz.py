import matplotlib.pyplot as plt
import numpy as np
import os

def get_num(line, startidx, sep1, sep2):
    idx1 = line[startidx:].find(sep1)
    idx2 = line[startidx + idx1 + 1:].find(sep2)
    return line[startidx + idx1 + 1:startidx + idx1 + 1 + idx2]

current_dir = os.path.dirname(os.path.realpath(__file__))
build_dir = os.path.abspath(os.path.join(current_dir, '../build'))


X_FILT = []
Y_FILT = []
Z_FILT = []

for filename in os.listdir(build_dir):
    if (filename == "results_filt.txt"):
        flog = os.path.abspath(os.path.join(build_dir, filename))
        with open(flog) as f:
            lines = f.readlines()

        FRAME = []
        X = []
        Y = []
        Z = []

        for line in lines:
            frameidx = line.find('Frame')
            xidx = line.find('x')
            yidx = line.find('y')
            zidx = line.find('z')

            frame = get_num(line, frameidx, ' ', ',')
            x = get_num(line, xidx, ' ', ',')
            y = get_num(line, yidx, ' ', ',')
            z = get_num(line, zidx, ' ', ';')

            FRAME.append(int(frame))
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))

        X_FILT = X
        Y_FILT = Y
        Z_FILT = Z

    else:
        continue

#plt.style.use('dark_background')

fig = plt.figure('Results', figsize=(20, 14))

plt.plot(X, color='orange', linewidth=1.5, label='X')
plt.plot(Y, color='blue', linewidth=1.5, label='Y')
plt.plot(Z, color='green', linewidth=1.5, label='Z')
plt.title('Difference in translation', fontsize=15)
plt.xlabel('frames', fontsize=15)
plt.ylabel('coords', fontsize=15)
plt.grid(True)
plt.legend()

fig.tight_layout()
plt.savefig('results_plot.png', dpi=300)
plt.show()

