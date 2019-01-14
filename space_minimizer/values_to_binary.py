#!/usr/bin/python3

import binaryspace
from binaryspace import BinarySpace

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        default="",
                        required=True)
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument('--space_size',
                        type=int,
                        default=255,
                        required=False)
    parser.add_argument('--niter',
                        type=int,
                        default=100,
                        required=False)
    args = parser.parse_args()


    bs = BinarySpace(args.name + '_' + str(int(args.alpha * 10)), 8100, gravity_alpha=args.alpha, end_count=args.niter)
    g = bs.create_graph_equal_spacing(args.space_size)
    bs.minimize_energy_given_graph(g)
