import numpy as np

with open("201901121508-pbox.biasedEGreedyFullLabelling2.kwy") as a:
    a = a.read().strip().split('\n')

with open("201901131720-pbox.biasedEGreedyFullLabelling25.kwy") as b:
    b = b.read().strip().split('\n')

a = a[11:]
b = b[11:]

a = [[int(el) if el != 'o' and el != 't' else el for el in l.split('\t')] for l in a]

b = [[int(el) if el != 'o' and el != 't' else el for el in l.split('\t')] for l in b]

splice_point = a[-1]

start_index = splice_point[0]
start_time = splice_point[1]
end_time = splice_point[2]

for entry in b:
    entry[0] += start_index
    entry[1] += start_time
    entry[2] += end_time
    a.append(entry)
    
a = [[str(e) for e in entry] for entry in a]
print(len(a))
print(a[:10])
print()
print(a[-10:])

out_text = """# Keepers: 3
# Takers:  2
# Region:  20 x 20
#
# Description of Fields:
# 1) Episode number
# 2) Start time in simulator steps (100ms)
# 3) End time in simulator steps (100ms)
# 4) Duration in simulator steps (100ms)
# 5) (o)ut of bounds / (t)aken away
#
"""

with open("out_log.txt", 'w') as outfile:
    outfile.write(out_text)
    for entry in a:
        outfile.write("\t".join(entry))
        outfile.write('\n')

