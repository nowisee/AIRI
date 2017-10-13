#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import
import numpy as np

N_JOINTS = 16

LINKS = (
    (7, 3), (3, 2), (2, 1), # 골반에서 우측 발까지
    (7, 4), (4, 5), (5, 6), # 골반에서 좌측 발까지
    (7, 8), (8, 9), (9, 10), # 골반에서 정수리까지
    (8, 13), (13, 12), (12, 11), # 명치에서 우측 손 끝 까지
    (8, 14), (14, 15), (15, 16), # 명치에서 좌측 손 끝 까지
)


def motion_to_jrel2_row(row):
    row = row[:3*N_JOINTS].reshape([-1,3])
    row_rel = np.zeros_like(row)
    j1, j2 = LINKS[0]
    row_rel[j1-1] = row[j1-1]
    for j1, j2 in LINKS:
        row_rel[j2-1] = row[j2-1] - row[j1-1]
    row = row_rel.reshape([-1])
    return row


def motion_to_jrel2(data):
    assert data.ndim == 2, ('unexpected data.ndim != 2',data.ndim)

    for r in range(len(data)):
        data[r,:] = motion_to_jrel2_row(data[r,:])

    return data


def jrel2_to_motion_row(row):
    row = row[:3*N_JOINTS].reshape([-1,3])
    for j1, j2 in LINKS:
        row[j2-1] = row[j2-1] + row[j1-1]
    row = row.reshape([-1])
    return row


def jrel2_to_motion(data):
    assert data.ndim == 2, ('unexpected data.ndim != 2',data.ndim)

    for r in range(len(data)):
        data[r,:] = jrel2_to_motion_row(data[r,:])

    return data


if __name__ == '__main__':
    import os, sys

    
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file')
        parser.add_argument('output_file')
        parser.add_argument('-r','--reverse',action='store_const',const=True,default=False)
        args = parser.parse_args()
        print(vars(args),file=sys.stderr)
        return args

    
    args           = parse_args()
    input_file     = args.input_file
    output_file    = args.output_file

    try: os.makedirs(os.path.dirname(output_file))
    except: pass

    data           = np.loadtxt(input_file, delimiter=',')
    if args.reverse:
        data       = jrel2_to_motion(data)
    else:
        data       = motion_to_jrel2(data)
    np.savetxt(output_file, data, fmt='%f', delimiter=',')
