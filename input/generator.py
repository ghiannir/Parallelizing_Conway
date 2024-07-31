from random import randint

N = 1000

fp = open("input.txt", "w")

for i in range(N):
    for j in range(N):
        x = randint(0, 1)
        if(x == 0):
            fp.write("O")
        else:
            fp.write("X")
    fp.write("\n")