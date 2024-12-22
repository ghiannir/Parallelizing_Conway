import random
import sys
from os import environ

try:
    N = int(environ.get('N'))
    DIS = float(environ.get('DIS'))
except:
    N = int(sys.argv[1])
    DIS = 0.5

filename = "../input/input_"+sys.argv[2]+".txt"

fp = open(filename, 'w')

for i in range(N):
    sequence = [0 if random.random() < DIS else 1 for _ in range(N)]
    separator = ' '
    str_sequence = [str(n) for n in sequence]
    sequence = separator.join(str_sequence)
    fp.write(str(sequence))
    fp.write("\n")
fp.close()