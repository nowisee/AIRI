#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division, absolute_import
import numpy as np

N_JOINTS = 16

def motion_to_jointrel(data):
    assert data.ndim == 2, ('unexpected data.ndim != 2',data.ndim)

    # split joints, make 3-D coords
    joints            = data[:,:].reshape([-1,N_JOINTS,3])

    # joint-relative
    j7                = joints[:,6:7,:]  # center of body
    joints[:,0:6,:]  -= j7
    joints[:,7:,:]   -= j7
    data[:,:N_JOINTS*3] = joints.reshape([-1,N_JOINTS*3])
    del joints

    return data


def jointrel_to_motion(data):
    assert data.ndim == 2, ('unexpected data.ndim != 2',data.ndim)

    # split joints, make 3-D coords
    joints            = data[:,:].reshape([-1,N_JOINTS,3])

    # joint-relative
    j7                = joints[:,6:7,:]  # center of body
    joints[:,0:6,:]  += j7
    joints[:,7:,:]   += j7
    data[:,:N_JOINTS*3] = joints.reshape([-1,N_JOINTS*3])
    del joints

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
        data       = jointrel_to_motion(data)
    else:
        data       = motion_to_jointrel(data)
    np.savetxt(output_file, data, fmt='%.7f', delimiter=',')
