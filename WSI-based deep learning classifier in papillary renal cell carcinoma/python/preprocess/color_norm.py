#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from PIL import Image
import glob
import random
import sys
import getopt
import cupy as cp
import tifffile as tif
import cv2
import gc
from tqdm import tqdm


class HENormalizer:
    def fit(self, target):
        pass

    def normalize(self, I, **kwargs):
        raise Exception('Abstract method')

"""
Inspired by torchstain :
Source code adapted from: https://github.com/schaugf/HEnorm_python;
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class Normalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = cp.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = cp.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -cp.log((I.astype(cp.float32)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~cp.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = cp.arctan2(That[:,1],That[:,0])

        minPhi = cp.percentile(phi, alpha)
        maxPhi = cp.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(cp.array([(cp.cos(minPhi), cp.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(cp.array([(cp.cos(maxPhi), cp.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = cp.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = cp.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = cp.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = cp.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = cp.array([cp.percentile(C[0,:], 99), cp.percentile(C[1,:],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        I = cp.asarray(I)
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15):
        ''' Normalize staining appearence of H&E stained images
        Example use:
            see test.py
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity
        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image
        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
       # I = cp.asarray(I)
        batch,h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = cp.divide(maxC, self.maxCRef)
        C2 = cp.divide(C, maxC[:, cp.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = cp.multiply(Io, cp.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = cp.reshape(Inorm.T, (batch,h, w, c)).astype(cp.uint8)



        return Inorm

all_centers = None
centers =  None
valid_center =  None
centers_num = None
TILE_SIZE = None
SEED = 22

df_path = Path(None)

argv = sys.argv[1:]  # Params used in batch processing
opts,args = getopt.getopt(argv,'s:l:i:n:')
for opt,arg in opts:
    if opt in ['-s']:
        subset = arg
    elif opt in ['-l']:
        LEVEL= arg
    elif opt in ["-i"]:
        INDEX=int(arg)
    elif opt in ["-n"]:
        n = int(arg)

"""adjust case numbers"""
cases_np = np.load(None,allow_pickle=True) #load pre-sampled tiles path array

cases_np = cases_np[n*(INDEX-1):n*INDEX]

classes_to_index = {'nonrecurrence':0,'recurrence':1}


def _bytes_feature(value):
    """
        Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """
    Returns a float_list from a float / double.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def Write_TFrecords_casewise(cases,TFrecords_DIR=None,level=LEVEL,TILE_SIZE=512):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    templates = tif.imread(list(np.load("templates.npy",allow_pickle=True)))#load templates
    normalizer = Normalizer()
    normalizer.fit(templates)
    for case in tqdm(cases_np):
        case_name = Path(case[0][0]).parent.parent.stem
        cls = Path(case[0][0]).parent.parent.parent.stem
        label = classes_to_index[cls]
        tfrecords_file_path = Path(TFrecords_DIR)/level/f"{case_name}.tfrecords"
        if not tfrecords_file_path.exists():
            writer = tf.io.TFRecordWriter(str(tfrecords_file_path))
            for i in range(case.shape[0]):
                try:
                    tiles = tif.imread(list(case[i,:]))
                    imgs = cp.asarray(tiles)
                    norm_imgs= cp.asnumpy(normalizer.normalize(I=imgs))
                    img_data = norm_imgs.tobytes()
                    example = tf.train.Example(features = tf.train.Features(
                            feature={
                                "img_data":_bytes_feature(img_data),
                                "label":_int64_feature(label)
                            }))
                    writer.write(example.SerializeToString())

                except Exception as e:
                        print(e)
                        print(case_name,i)
                norm_imgs = None
                imgs = None
                mempool.free_all_blocks() 
                pinned_mempool.free_all_blocks()
                gc.collect()
            writer.close()

Write_TFrecords_casewise(cases_np,TFrecords_DIR=None,level=LEVEL,TILE_SIZE=TILE_SIZE)
