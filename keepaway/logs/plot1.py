from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
import sys
import os

def smooth(y, box_pts): # box_pts is how wide the convolution window is
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_variance(y, width):
    all_pts = []
    length = len(y)
    half = int(width//2)
    for i,_ in enumerate(y):
        if i < half or i > length - half:
            all_pts.append(0)
        else: # calculate the variance
            pts = np.array(y[i - half:i + half])
            v = pts.var()
            all_pts.append(v)
    return all_pts

def get_variance_one_pt(i, y, width):
    length = len(y)
    half = int(width//2)

    if i < half or i > length - half:
        return 0
    else: # calculate the variance
        pts = np.array(y[i - half:i + half])
        return pts.var()


# returns two points to plot, upper and lower bound
def get_middle_95(i, y, window_size):
    length = len(y)
    half = int(window_size//2)
    # only do if we're not too close to the edges
    if (i - half in range(length)) and (i + half in range(length)):
        pts = y[i - half:i + half]
        pts.sort() # order all points
        #print(int(window_size * 0.025))
        #print(int(window_size * 0.975))
        middle = pts[int(window_size * 0.025) : int(window_size * 0.975)] # remove top and bottom 2.5%
        biggest = max(middle)
        smallest = min(middle)
        print(biggest, smallest)
        return (biggest, smallest)
    else:
        return (-1, -1)


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


def plot_smooth(values, amount=1, colour=None, transp=None, legend=False):
    x, y, title, argumentation = values
    #fig, ax = plt.subplots(figsize=(14, 7))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    window_size = 900 * np.sqrt(amount)
    half = int(window_size//2)
    if colour and transp:
        plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=r"SARSA($\lambda$) with Argumentation", color=colour, alpha=transp)
    else:
        plt.plot(x[half:-half], smooth(y, window_size)[half:-half], 'b-', lw = 2, label=r"SARSA($\lambda$) with Argumentation")

    #if legend:
        #plt.legend(loc=4)
    #plt.xlim(right=np.max(x))
    #plt.title("SMDP Sarsa(lambda) {}, 3v2, 20x20 {}".format(argumentation, title))
    #plt.xlabel("Learning Time (hours)")
    #plt.ylabel("Episode Duration (seconds)")
    #plt.ylim((4, 17))
    #plt.xlim((0, 40))
    #plt.grid()
    #plt.style.use("classic") 


# returns quantities in seconds and hours
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


# returns quantities in seconds and hours
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


def plot_average(the_dir):
    sorted_tt, sorted_ep, a, b, maxval = prepare_average(the_dir)
    plot_smooth((sorted_tt, sorted_ep, a, b), amount=maxval, legend=True)


def plot_confidence_intervals(the_dir):
    sorted_tt, sorted_ep, a, b, maxval = prepare_average(the_dir)
    #print("t hours:ep duration")
    #for i in range(20):
        #print("{0:1.4f}:{1}".format(sorted_tt[i], sorted_ep[i]))

    upper_pts = []
    lower_pts = []
    xs = []
    window_size = 100000
    for index,_ in enumerate(sorted_ep):
        if index % 1000 != 0:
            continue
        upper, lower = get_middle_95(index, sorted_ep, window_size)
        if upper != -1:
            upper_pts.append(upper)
            lower_pts.append(lower)
            xs.append(index / 2000)

    plt.plot(xs, upper_pts, 'o', color = "red", alpha=0.2)
    plt.plot(xs, lower_pts, 'o', color = "red", alpha=0.2)


def plot_mean_confidence_interval(the_dir):
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

    plt.plot(spaced_tt, spaced_means + 1.96*spaced_variances, spaced_tt, spaced_means - 1.96*spaced_variances, ':', color = 'blue', alpha=0.2)
    plt.fill_between(spaced_tt, spaced_means + 1.96*spaced_variances, spaced_means - 1.96*spaced_variances, facecolor='blue', alpha=0.2, interpolate=True)

class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch


if __name__ == "__main__":


    # here we plot the 95% CI
    set_up_plot()
    #plot_confidence_intervals(sys.argv[1])
    plot_average(sys.argv[1])
    plot_mean_confidence_interval(sys.argv[1])

    #plot_average(sys.argv[1])
    # set up the plot

    # here we plot the individual lines
    ''' 
    for i,my_file in enumerate(os.listdir(sys.argv[1])):
        if ".kwy" in my_file:
            to_plot = extract_data(my_file, sys.argv[1])
            plot_smooth(to_plot, colour="blue", transp=0.2) # data, colour, transparent
    arg = "with Argumentation" if len(sys.argv) > 2 else ""
    '''

    bg = np.array([1, 1, 1])  # background of the legend is white
    colors = ['black', 'blue', 'green']
    # with alpha = .5, the faded color is the average of the background and color
    colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]

    plt.legend([0, 1, 2], ['Data 0', 'Data 1', 'Data 2'],
               handler_map={
                   0: LegendObject(colors[0], colors_faded[0]),
                   1: LegendObject(colors[1], colors_faded[1]),
                   2: LegendObject(colors[2], colors_faded[2], dashed=True),
                })
    
    #set_up_plot()
    
    plt.show()
