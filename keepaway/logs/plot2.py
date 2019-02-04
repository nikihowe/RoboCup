from matplotlib import pyplot as plt
import numpy as np
import sys
import os

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def set_up_plot(argumentation=""):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.title(r"SMDP SARSA($\lambda$), 3 vs 2 players, 20 x 20 field")
    plt.xlabel("Learning Time (hours)")
    plt.ylabel("Episode Duration (seconds)")
    plt.ylim((4, 17))
    plt.xlim((0, 40))
    plt.grid()
    plt.style.use("classic") 

def plot_smooth(values, values2=None, amount=1, amount2=1):
    x, y, title, argumentation = values
    #fig, ax = plt.subplots(figsize=(14, 7))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    window_size = 900 * np.sqrt(amount)
    half = int(window_size//2)
    plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=r"SARSA($\lambda$) with Argumentation")
    if values2:
        xx, yy, t, a = values2
        window_size2 = 900 * np.sqrt(amount2)
        half2 = int(window_size2//2)
        plt.plot(xx[half2:-half2], smooth(yy, window_size2)[half2:-half2], 'r-', lw = 2, label=r"SARSA($\lambda$)")
    plt.legend(loc=4)
    #plt.xlim(right=np.max(x))
    #plt.title("SMDP Sarsa(lambda) {}, 3v2, 20x20 {}".format(argumentation, title))
    #plt.xlabel("Learning Time (hours)")
    #plt.ylabel("Episode Duration (seconds)")
    #plt.ylim((4, 17))
    #plt.xlim((0, 40))
    #plt.grid()
    #plt.style.use("classic") 

def extract_data(filename, the_dir):
    with open(the_dir + "/" + filename) as afile:
        a = afile.read().strip().split('\n')

    argumentation = "with Argumentation" if len(sys.argv) > 2 else ""

    a = a[11:] # get rid of header

    a = [[int(el) if el != 'o' and el != 't' else el for el in l.split('\t')] for l in a]
    a = np.array(a)

    training_time = a[:,2].astype(np.float)/36000
    episode_duration = a[:,3].astype(np.float)/10
    taken_or_out = a[:,4]
    return (training_time, episode_duration, filename, argumentation)

def prepare_average(the_dir):
    training_times = []
    episode_durations = []
    for i, my_file in enumerate(os.listdir(the_dir)):
        if ".kwy" in my_file:
            tt, ed, fi, ar = extract_data(my_file, the_dir)
            training_times = np.concatenate((training_times, tt), axis=None)
            episode_durations = np.concatenate((episode_durations, ed), axis=None)
            print(my_file)
    maxval = len(os.listdir(the_dir))

    # Now take average
    my_pairs = []
    for i, el in enumerate(training_times):
        my_pairs.append((el, episode_durations[i]))
    my_pairs.sort()
    sorted_tt = []
    sorted_ep = []
    for a, b in my_pairs:
        sorted_tt.append(a)
        sorted_ep.append(b)
    return (sorted_tt, sorted_ep, "hi", "ho", maxval)

def plot_average(the_dir, the_dir2 = None):
    sorted_tt, sorted_ep, a, b, maxval = prepare_average(the_dir)
    if the_dir2:
        stt2, sep2, a2, b2, maxval2 = prepare_average(the_dir2)
        plot_smooth((sorted_tt, sorted_ep, a, b),
                (stt2, sep2, a2, b2), amount=maxval, amount2 = maxval2)
    else:
        plot_smooth((sorted_tt, sorted_ep, a, b), amount=maxval)

if __name__ == "__main__":
    '''
    #plot_average(sys.argv[1])
    set_up_plot()
    for i,my_file in enumerate(os.listdir(sys.argv[1])):
        if ".kwy" in my_file:
            to_plot = extract_data(my_file, sys.argv[1])
            plot_smooth(to_plot)
    plt.show()
    arg = "with Argumentation" if len(sys.argv) > 2 else ""
    '''
    set_up_plot()
    
    if len(sys.argv) > 2:
        plot_average(sys.argv[1], sys.argv[2])
    else:
        plot_average(sys.argv[1])
    plt.show()
