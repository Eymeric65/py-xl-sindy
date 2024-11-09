import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def Animate_Single_pendulum(L, q_v, t_v):

    x = L* np.sin(q_v[:, 0])
    y = -L * np.cos(q_v[:, 0])


    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
    ax.set_aspect('equal')
    ax.grid()

    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x[i]]
        thisy = [0, y[i]]

        history_x = x[:i]
        history_y = y[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (t_v[i]))
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(q_v), interval=40, blit=True)
    plt.show()

# Bricolage sans class
def Single_pendulum_one_state(figure,L,q,t):

    x = L* np.sin(q[0])
    y = -L * np.cos(q[0])

    ax = figure.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([0,x], [0,y], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, time_template % t, transform=ax.transAxes)

    return ax,line,time_text

def Single_pendulum_update(line,time_text,L,q,t):

    x = L* np.sin(q[0])
    y = -L * np.cos(q[0])

    line.set_data([0,x], [0,y])
    time_template = 'time = %.1fs'
    time_text.set_text(time_template % (t))
#------------------------

def Animate_double_pendulum(L1, L2, q_v, t_v):
    Lt = L1 + L2

    x1 = L1 * np.sin(q_v[:, 0])
    y1 = -L1 * np.cos(q_v[:, 0])

    x2 = L2 * np.sin(q_v[:, 2]) + x1
    y2 = -L2 * np.cos(q_v[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-Lt, Lt), ylim=(-Lt, Lt))
    ax.set_aspect('equal')
    ax.grid()

    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (t_v[i]))
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(q_v), interval=40, blit=True)
    plt.show()

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '*' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
