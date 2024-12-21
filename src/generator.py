import random
from os import environ
N = int(environ.get('N'))
DIS = float(environ.get('DIS'))

fp = open("../input/input.txt", "w")

for i in range(N):
    sequence = [0 if random.random() < DIS else 1 for _ in range(N)]
    separator = ' '
    str_sequence = [str(n) for n in sequence]
    sequence = separator.join(str_sequence)
    fp.write(str(sequence))
    fp.write("\n")
fp.close()