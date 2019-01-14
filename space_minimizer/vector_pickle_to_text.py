#!/usr/bin/python3

import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        type=str,
                        default="",
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        default="",
                        required=True)
    args = parser.parse_args()


    with open(args.i, "rb") as input_file:
        e = pickle.load(input_file)

    X = e[1]

    with open(args.o, 'a') as the_file:
        count = 0

        for i in X:
            line = ''

            for j in range(len(i.vector[0])):
                if i.vector[0][j]:
                    line = line + '1'
                else:
                    line = line + '0'

            #print('Line ' + str(count) + ') ' + line)

            count += 1

            the_file.write(line + '\n')
