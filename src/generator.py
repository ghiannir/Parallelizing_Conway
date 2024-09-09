from random import randint
from os import environ

N = int(environ.get('N'))

fp = open("../input/input.txt", "w")

for i in range(N):
    for j in range(N):
        x = randint(0, 1)
        if(x == 0):
            fp.write("1")
        else:
            fp.write("0")
        fp.write(" ")
    fp.write("\n")
fp.close()