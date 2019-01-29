import sys

alpha = float(sys.argv[2])
coarse = int(sys.argv[3])
window_size = int(sys.argv[1])
q = []
my_sum = 0

my_file = sys.stdin
for i in range(11):
    my_file.readline()

for i in range(window_size):
    this_run = my_file.readline().split('\t')
    q.append(int(this_run[3]))
    my_sum += q[i]

i = 0
ccount = 0

start = 0
prev = sum

while (True):

