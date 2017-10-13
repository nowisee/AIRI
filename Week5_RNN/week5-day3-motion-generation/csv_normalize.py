#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import
import sys
import numpy as np

N_JOINTS = 16


def normalize(input_file,mean,std,output_file,scale=1.0):
    data  = np.loadtxt(input_file, delimiter=',')
    data -= mean
    data *= scale
    data /= std
    np.savetxt(output_file, data, fmt='%f', delimiter=',')
    print('normalize:', input_file, output_file, file=sys.stderr)
    return len(data), np.mean(data,0), np.var(data,0), None, None


def unnormalize(input_file,mean,std,output_file,scale=1.0):
    data  = np.loadtxt(input_file, delimiter=',')
    data *= std
    data /= scale
    data += mean
    np.savetxt(output_file, data, fmt='%f', delimiter=',')
    print('un-normalize:', input_file, output_file, file=sys.stderr)
    return len(data), np.mean(data,0), np.var(data,0), None, None


if __name__ == '__main__':
    import os
    import pandas as pd


    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file')
        parser.add_argument('output_file')
        parser.add_argument('-m','--mean_file',required=True)
        parser.add_argument('-s','--std_file',required=True)
        parser.add_argument('-x','--scale',type=float,default=1.0)
        parser.add_argument('-r','--reverse',action='store_const',const=True,default=False)
        parser.add_argument('-v','--verbose',action='store_const',const=True,default=False)
        args = parser.parse_args()
        print(vars(args), file=sys.stderr)
        return args


    args = parse_args()

    mean = np.loadtxt(args.mean_file, delimiter=',')
    std  = np.loadtxt(args.std_file, delimiter=',')

    if args.verbose:
        print('=================================', file=sys.stderr)
        print('input count:', count, file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('mean:', file=sys.stderr)
        print(pd.Series(mean[0]).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('std:', file=sys.stderr)
        print(pd.Series(std[0]).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)

    try: os.makedirs(os.path.dirname(args.output_file))
    except: pass

    if args.reverse:
        _, mean_, var_, _, _ = \
            unnormalize(
                args.input_file, mean, std, args.output_file, args.scale)
    else:
        _, mean_, var_, _, _ = \
            normalize(
                args.input_file, mean, std, args.output_file, args.scale)

    std_ = np.sqrt(var_)

    if args.verbose:
        print('=================================', file=sys.stderr)
        print('output count:', count, file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('mean:', file=sys.stderr)
        print(pd.Series(mean_[0]).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('std:', file=sys.stderr)
        print(pd.Series(std_[0]).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)
