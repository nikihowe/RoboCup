import subprocess
import os
import sys

my_dir = sys.argv[1]

os.chdir(my_dir)

for i,filename in enumerate(os.listdir('.')):
    print(filename)
    subprocess.call(['cat {} | ../winsum 900 0.01 30 > {}.out'.format(filename, i)],shell=True)
