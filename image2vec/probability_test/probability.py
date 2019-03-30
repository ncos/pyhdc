#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, signal, math, time


def P(nbits, prob_list):
    p = prob_list[0]
    if (len(prob_list) == 1):
        if (nbits > 1): return 0.0
        if (nbits == 1): return p
        if (nbits <= 0): return 1.0
    return (1 - p) * P(nbits, prob_list[1:]) + p * P(nbits - 1, prob_list[1:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size',
                        type=int)
    parser.add_argument('--prob',
                        type=float,
                        required=False,
                        default=0.0)
    parser.add_argument('--th',
                        type=float,
                        required=False,
                        default=0.5)
    args = parser.parse_args()

    prob_list = []
    prob_list.append(args.prob)

    for i in range(args.size - 1):
        prob_list.append(0.5)

    nbits = int(math.ceil(args.size * args.th))

    print ("Probabilities:")
    print (prob_list)
    print ("nbits:", nbits)
    print ()

    print(P(nbits, prob_list)) 
