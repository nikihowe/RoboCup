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
    plt.ylim((4, 20))
    plt.xlim((0, 40))
    plt.grid()
    plt.style.use("classic") 

def get_variance_one_pt(i, y, width):
    length = len(y)
    half = int(width//2)

    if i < half or i > length - half:
        return 0
    else: # calculate the variance
        pts = np.array(y[i - half:i + half])
        return pts.var()

def plot_smooth(values, amount=1, colour=None, transp=None, legend=False):
    x, y, title, argumentation = values
    window_size = 900 * np.sqrt(amount)
    half = int(window_size//2)
    if colour and transp:
        plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=None, color=colour, alpha=transp)
    else:
        plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=None)

def plot_smooth3(values, values2, values3, amount=1, amount2=1, amount3=1, legend=False):
    x, y, title, argumentation = values
    window_size = 900 * np.sqrt(amount)
    half = int(window_size//2)
    plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=r"SARSA($\lambda$) with Argumentation")

    xx, yy, t, a = values2
    window_size2 = 900 * np.sqrt(amount2)
    half2 = int(window_size2//2)
    plt.plot(xx[half2:-half2], smooth(yy, window_size2)[half2:-half2], 'g-', lw = 2, label=r"SARSA($\lambda$)")

    xx, yy, t, a = values3
    window_size3 = 900 * np.sqrt(amount3)
    half3 = int(window_size3//2)
    plt.plot(xx[half3:-half3], smooth(yy, window_size3)[half3:-half3], 'r-', lw = 2, label=r"SARSA($\lambda$)")


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
    dir_size = 0
    for i, my_file in enumerate(os.listdir(the_dir)):
        if ".kwy" in my_file:
            dir_size += 1
            tt, ed, fi, ar = extract_data(my_file, the_dir)
            training_times = np.concatenate((training_times, tt), axis=None)
            episode_durations = np.concatenate((episode_durations, ed), axis=None)
            print(my_file)

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
    return (sorted_tt, sorted_ep, "hi", "ho", dir_size)

def plot_average(the_dir1, the_dir2, the_dir3):
    sorted_tt, sorted_ep, a, b, maxval = prepare_average(the_dir1)
    stt2, sep2, a2, b2, maxval2 = prepare_average(the_dir2)
    stt3, sep3, a3, b3, maxval3 = prepare_average(the_dir3)
    plot_smooth3(
        (sorted_tt, sorted_ep, a, b),
        (stt2, sep2, a2, b2),
        (stt3, sep3, a3, b3),
        amount=maxval,
        amount2 = maxval2,
        amount3 = maxval3,
        legend=True
    )

def plot_mean_confidence_interval(the_dir, c):
    sorted_tt, sorted_ep, a, b, maxval = prepare_average(the_dir)
    sorted_tt = np.array(sorted_tt)
    sorted_ep = np.array(sorted_ep)
    upper_pts = []
    lower_pts = []
    xs = []
    
    window_size = 900 * np.sqrt(maxval)
    half = int(window_size // 2)

    #variances = np.array(get_variance(sorted_ep, window_size))
    means = np.array(smooth(sorted_ep, window_size))

    spaced_means = []
    spaced_variances = []
    spaced_tt = []
    for i,_ in enumerate(sorted_ep):
        if i % 100 == 0:
            spaced_tt.append(sorted_tt[i])
            spaced_means.append(means[i])
            v = get_variance_one_pt(i, sorted_ep, window_size)
            std_error = np.sqrt(v/window_size)
            spaced_variances.append(std_error)
    spaced_means = np.array(spaced_means)
    spaced_variances = np.array(spaced_variances)

    half = int(half/100)
    spaced_means = spaced_means[half:-half]
    spaced_variances = spaced_variances[half:-half]
    spaced_tt = spaced_tt[half:-half]

    plt.plot(spaced_tt, spaced_means + 1.96*spaced_variances, spaced_tt, spaced_means - 1.96*spaced_variances, ':', color = c, alpha=0.2)
    plt.fill_between(spaced_tt, spaced_means + 1.96*spaced_variances, spaced_means - 1.96*spaced_variances, facecolor=c, alpha=0.2, interpolate=True)

if __name__ == "__main__":
    
    set_up_plot()

    '''
    for i,my_file in enumerate(os.listdir(sys.argv[1])):
        if ".kwy" in my_file:
            to_plot = extract_data(my_file, sys.argv[1])
            plot_smooth(to_plot, colour="blue", transp=0.2) # data, colour, transparent

    for i,my_file in enumerate(os.listdir(sys.argv[2])):
        if ".kwy" in my_file:
            to_plot = extract_data(my_file, sys.argv[2])
            plot_smooth(to_plot, colour="green", transp=0.2)

    for i,my_file in enumerate(os.listdir(sys.argv[3])):
        if ".kwy" in my_file:
            to_plot = extract_data(my_file, sys.argv[3])
            plot_smooth(to_plot, colour="red", transp=0.2)
    '''

    arg = "with Argumentation"
    
    plot_average(sys.argv[1], sys.argv[2], sys.argv[3])
    plot_mean_confidence_interval(sys.argv[1], 'blue')
    plot_mean_confidence_interval(sys.argv[2], 'green')
    plot_mean_confidence_interval(sys.argv[3], 'red')
    #plt.legend()
    plt.show()
