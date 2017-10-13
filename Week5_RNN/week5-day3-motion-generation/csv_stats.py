#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import
import numpy as np

class CsvStats():
    """
    Cumulative Csv Stats

        CsvStats(self, ddof=0)

    about ddof=0, from numpy.var() documentation:

        The mean is normally calculated as x.sum() / N, where N = len(x). If,
        however, ddof is specified, the divisor N - ddof is used instead.
        In standard statistical practice, ddof=1 provides an unbiased
        estimator of the variance of a hypothetical infinite population.
        ddof=0 provides a maximum likelihood estimate of the variance
        for normally distributed variables.

    Usage:

        stats = CsvStats()
        for s in samples:
            stats.append(
                count=s.shape[0],  # count
                mean=np.mean(s, axis=0), # mean
                var=np.var(s, axis=0),   # var
                min=np.amin(s, axis=0),   # min
                max=np.amax(s, axis=0))

        print(stats.stats(), file=sys.stderr)

    """
    def __init__(self, ddof=0):
        self.count = 0
        self.mean  = 0
        self.var   = 0
        self.min   = None
        self.max   = None
        self.ddof  = ddof


    def stats(self):
        return self.count, self.mean, self.var, self.min, self.max


    def append(self, count, mean, var, min_=None, max_=None):
        # new average
        new_count    = self.count + count
        new_mean     = (self.count * self.mean + count * mean) / new_count

        # new variance
        delta        = mean - self.mean
        m_a          = self.var * (self.count - self.ddof)
        m_b          = var * (count - self.ddof)
        m2           = m_a + m_b + delta * delta * self.count * count / new_count
        new_var      = m2 / (new_count - self.ddof)

        # new min
        if self.min is None:
            self.min = min_
        else:
            if min is not None:
                self.min = np.minimum(self.min, min_)

        # new max
        if self.max is None:
            self.max = max_
        else:
            if max is not None:
                self.max = np.maximum(self.max, max_)

        self.count = new_count
        self.mean  = new_mean
        self.var   = new_var

        return self.stats()

    
    def append_data(self,data):
        count = data.shape[0]
        mean  = np.mean(data,axis=0)
        var   = np.var(data,axis=0,ddof=self.ddof)
        min_   = np.amin(data,axis=0)
        max_   = np.amax(data,axis=0)
        return self.append(count, mean, var, min_=min_, max_=max_)


if __name__ == '__main__':
    """
    unit test
    """
    import os
    import sys
    import pandas as pd
    
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input_files',nargs='*')
        parser.add_argument('-m','--mean_file')
        parser.add_argument('-s','--std_file')
        parser.add_argument('-v','--var_file')
        parser.add_argument('--verbose',action='store_const',const=True,default=False)
        args = parser.parse_args()
        print(vars(args), file=sys.stderr)
        return args

    def self_test():
        samples = [
            np.asarray(np.random.uniform(-0.5,999.0,[np.random.randint(10,300), 800]),dtype=np.float64)
            for _ in range(200)]

        stats = CsvStats()
        for s in samples:
            stats.append_data(s)

        # print(stats.stats(), file=sys.stderr)

        # compare with whole result

        merged = np.concatenate(samples,axis=0)
        count  = merged.shape[0]
        mean   = np.mean(merged, axis=0)
        var    = np.var(merged, axis=0)
        min_    = np.min(merged, axis=0)
        max_    = np.max(merged, axis=0)

        # print((count,mean,var,min_,max_), file=sys.stderr)

        # check squared errors

        for a, b in zip(stats.stats(), (count,mean,var,min_,max_)):
            print(np.amax(np.square(a-b)), file=sys.stderr)


    args = parse_args()
    
    if 0 == len(args.input_files):
        self_test()
        sys.exit(0)

    stats = CsvStats(ddof=1)
    
    for fn in args.input_files:
        print('input:', fn, file=sys.stderr)
        data = np.loadtxt(fn, delimiter=',')
        stats.append_data(data)

    count, mean, var, _, _ = stats.stats()

    std = np.sqrt(var)

    if args.mean_file:
        try: os.makedirs(os.path.dirname(args.mean_file))
        except: pass
        np.savetxt(args.mean_file, np.atleast_2d(mean), fmt='%f', delimiter=',')
        print('wrote:', args.mean_file, file=sys.stderr)
    if args.std_file:
        try: os.makedirs(os.path.dirname(args.std_file))
        except: pass
        np.savetxt(args.std_file, np.atleast_2d(std), fmt='%f', delimiter=',')
        print('wrote:', args.std_file, file=sys.stderr)
    if args.var_file:
        try: os.makedirs(os.path.dirname(args.var_file))
        except: pass
        np.savetxt(args.var_file, np.atleast_2d(var), fmt='%f', delimiter=',')
        print('wrote:', args.var_file, file=sys.stderr)

    if args.verbose:
        print('=================================', file=sys.stderr)
        print('count:', count, file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('mean:', file=sys.stderr)
        print(pd.Series(mean).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)
        print('std:', file=sys.stderr)
        print(pd.Series(std).describe(), file=sys.stderr)
        print('=================================', file=sys.stderr)
