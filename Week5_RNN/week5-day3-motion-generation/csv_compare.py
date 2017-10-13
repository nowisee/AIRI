#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import
import sys
import numpy as np

def compare(data1,data2):
    xse = np.max(np.square(data1-data2))
    print('max square error:',xse,file=sys.stderr)
    return xse

if __name__ == '__main__':

    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input1')
        parser.add_argument('input2')
        args = parser.parse_args()
        print(vars(args),file=sys.stderr)
        return args

    args = parse_args()
    data1 = np.loadtxt(args.input1,delimiter=',')
    data2 = np.loadtxt(args.input2,delimiter=',')
    compare(data1,data2)
