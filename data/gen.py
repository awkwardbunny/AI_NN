#!/usr/bin/env python3
import random
import sys

usage = '''Usage: ./program.py <# of input nodes> <# of hidden nodes> <# of output nodes>
       ./program.py <n_i> <n_h> <n_o>'''

if len(sys.argv) < 4:
    print(usage)
    sys.exit(-1)

# Print numbers of nodes given from argv
print(' '.join(sys.argv[1:4]))

nNodes = list(map(int, sys.argv[1:4]))
#print(nNodes)
for l in range(1, len(nNodes)):
    for i in range(nNodes[l]):
        for j in range(nNodes[l-1]+1): # +1 for bias weight
            print("{:.3f}".format(random.uniform(0,1)), end='')
            if j != nNodes[l-1]:
                print(" ", end='')
        print()
