from matplotlib import pyplot as plt
import numpy as np
import sys

with open(sys.argv[1]) as afile:
    a = afile.read().strip().split('\n')

hundreds = float(sys.argv[2]) if len(sys.argv) > 2 else 10e6
argumentation = "with Argumentation" if len(sys.argv) > 4 else ""

a = a[11:] # get rid of header

a = [[int(el) if el != 'o' and el != 't' else el for el in l.split('\t')] for l in a]
a = np.array(a)

training_time = a[:,2].astype(np.float)/36000
episode_duration = a[:,3].astype(np.float)/10
taken_or_out = a[:,4]

bin_width = int(sys.argv[3]) if len(sys.argv) > 3 else 300
btt = []
bed = []
t = []
count = 0
tt_tot = 0
ed_tot = 0
taken_tot = 0
for i,x in enumerate(training_time):
    if (x > hundreds): break
    count = (count + 1) % bin_width
    tt_tot += x
    ed_tot += episode_duration[i]
    if taken_or_out[i] == 't': taken_tot += 1
    if count == 0:
        btt.append(tt_tot/bin_width)
        bed.append(ed_tot/bin_width)
        t.append(taken_tot >= bin_width/2)
        tt_tot = 0
        ed_tot = 0
        taken_tot = 0

training_time = btt
episode_duration = bed

fig, ax = plt.subplots(figsize=(14, 7))
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.plot(training_time, episode_duration, 'o')
#plt.plot(np.unique(training_time),
        #np.poly1d(np.polyfit(training_time, episode_duration, 1))
        #(np.unique(training_time)), linewidth=4.0)
plt.xlim(right=np.max(training_time))
plt.title("SMDP Sarsa(lambda) {}, 3v2, 20x20".format(argumentation))
plt.xlabel("Learning Time (hours)")
plt.ylabel("Episode Duration (seconds)")
plt.ylim((4, 16))
plt.xlim((0, 100))
plt.grid()
plt.style.use("classic") 
# Set figure width to 12 and height to 9
plt.show()
