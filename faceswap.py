#!/usr/bin/env python3
import tensorflow as tf
from keras import backend as K

GPU = True
CPU = False
num_cores = 4

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                        inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

import argparse
from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from scripts.convert import ConvertImage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    extract = ExtractTrainingData(
        subparser, "extract", "Extract the faces from a pictures.")
    train = TrainingProcessor(
        subparser, "train", "This command trains the model for the two faces A and B.")
    convert = ConvertImage(
        subparser, "convert", "Convert a source image to a new one with the face swapped.")
    arguments = parser.parse_args()
    try:
        arguments.func(arguments)
    except:
        parser.print_help()
