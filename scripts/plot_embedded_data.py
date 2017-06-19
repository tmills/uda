#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt

def main(args):
    if len(args) < 2:
        sys.stderr.write("Two required arguments: <data> <colors>\n")
        sys.exit(-1)

    data1, data2 = np.load(args[0])
    colors1, colors2 = np.load(args[1])
    d1neg = np.where(colors1 == 1)[0]
    d1pos = np.where(colors1 == 2)[0]
    d2neg = np.where(colors2 == 3)[0]
    d2pos = np.where(colors2 == 4)[0]

    print("Size of domain 1=%d, positive=%d, negative=%d" % (data1.shape[0], len(d1pos), len(d1neg)))
    print("Size of domain 2=%d, positive=%d, negative=%d" % (data2.shape[0], len(d2pos), len(d2neg)))

    print("First plot is all data together (shape=corpus, color=label)")
    plt.scatter(data1[:,0], data1[:,1], c=colors1, marker='o')
    plt.scatter(data2[:,0], data2[:,1], c=colors2, marker='x')
    plt.title("All data from both domains (shape=corpus, color=label)")

    print("Second plot is all positive data points (o=domain 1, x=domain 2)")
    plt.figure(2)
    plt.scatter(data1[d1pos,0], data1[d1pos,1], c=colors1[d1pos], marker='o')
    plt.scatter(data2[d2pos,0], data2[d2pos,1], c=colors2[d2pos], marker='x')
    plt.title('All positive data from both domains (o=domain 1, x=domain 2)')
    #print("Plotting %d points from second dataset" % (data2.shape[0]))
    #plt.scatter(data2[:,0], data2[:,1], c=colors2, marker='x')

    print("Third plot is positive domain 1(o) vs. negative domain 2 (x) (e.g., potential recall errors)")
    plt.figure(3)
    plt.scatter(data1[d1pos,0], data1[d1pos,1], c=colors1[d1pos], marker='o')
    plt.scatter(data2[d2neg,0], data2[d2neg,1], c=colors2[d2neg]-2, marker='x')
    plt.title('Domain 1 positive vs. domain 2 negative (recall errors)')

    print("Fourth plot is negative domain 1 (o) vs. positive domain 2 (x) (e.g., potential precision errors)")
    plt.figure(4)
    plt.scatter(data1[d1neg,0], data1[d1neg,1], c=colors1[d1neg], marker='o')
    plt.scatter(data2[d2pos,0], data2[d2pos,1], c=colors2[d2pos]-2, marker='x')
    plt.title('Domain 1 negative vs. domain 2 positive (precision errors)')
    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
