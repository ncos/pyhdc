#!/usr/bin/python3

import os
# import sys
import argparse
import pyhdc
import cv2
import operator
import img2vec
import numpy
import math

from random import sample

# sys.path.append(sys.path[0])


def get_directories(parent_dir):
    output = [dI for dI in os.listdir(parent_dir) if
              os.path.isdir(os.path.join(parent_dir, dI))]

    return output


def get_dvs_images(parent_dir):
    output = [f for f in os.listdir(parent_dir) if
              os.path.isfile(os.path.join(parent_dir, f))]

    return output


def preprocess_default_tests(tests, others, vmap, maximages=None):
    processed_tests = {}
    processed_others = {}

    if maximages is None:
        sample_tests = tests.keys()
        sample_others = {}

        for other in others.keys():
            sample_others[other] = others[other].keys()
    else:
        sample_tests = sample(tests.keys(), min(maximages, len(tests.keys())))
        sample_others = {}

        for other in others.keys():
            sample_others[other] = \
                sample(others[other].keys(), min(maximages, len(others[other].keys())))

    for test in sample_tests:
        processed_tests[test] = img2vec.img2vec(tests[test], vmap)

    for other in sample_others.keys():
        processed_others[other] = {}

        for target in sample_others[other]:
            processed_others[other][target] = img2vec.img2vec(others[other][target], vmap)

    return [processed_tests, processed_others]


def preprocess_window_tests(tests, others, vmap, maximages=None, size=50, stride=50):
    processed_tests = {}
    processed_others = {}

    if maximages is None:
        sample_tests = tests.keys()
        sample_others = {}

        for other in others.keys():
            sample_others[other] = others[other].keys()
    else:
        sample_tests = sample(tests.keys(), min(maximages, len(tests.keys())))
        sample_others = {}

        for other in others.keys():
            sample_others[other] = \
                sample(others[other].keys(), min(maximages, len(others[other].keys())))

    for test in sample_tests:
        processed_tests[test] = window_stack(tests[test], vmap, size, stride)

    for other in sample_others.keys():
        processed_others[other] = {}

        for target in sample_others[other]:
            processed_others[other][target] = \
                window_stack(others[other][target], vmap, size, stride)

    return [processed_tests, processed_others]


def window_stack(image, vmap, size, stride):
    vectors = []

    if not (image.shape[0] - size) % stride == 0 or not (image.shape[1] - size) % stride == 0:
        x = numpy.zeros((int((math.ceil((image.shape[0] - size) / stride) +
                              int(math.ceil(size / stride))) * stride),
                         int((math.ceil((image.shape[1] - size) / stride) +
                              int(math.ceil(size / stride))) * stride),
                         image.shape[2]),
                        dtype=int)
        x[:image.shape[0], :image.shape[1], :] = image
        image = x

        # print('New image size: ' + str(image.shape))

    # print('Image size: ' + str(image.shape))

    row = 0

    while row + size < image.shape[0]:
        # print('\trow: ' + str(row))

        col = 0

        while col + size < image.shape[1]:
            # print('\t\tcol: ' + str(col))

            window = img2vec.img2vec(image[row:row + size, col:col + size, :], vmap)
            vectors.append(window)

            col += stride

        row += stride

    # print('DONE---')

    ret = pyhdc.LBV()

    if len(vectors) == 1:
        ret.xor(vectors[0])

        return ret

    th = len(vectors) // 2

    for i in range(pyhdc.get_vector_width()):
        cnt = 0

        for v in vectors:
            if v.get_bit(i):
                cnt += 1

        if cnt >= th:
            ret.flip(i)

    return ret


class EmbeddingTester:

    main_dir = None
    to_test = None
    vmap = None
    test = 'default'
    test_types = {'default', 'window'}
    maximages = None
    size = 50
    stride = 50

    def __init__(self, main_dir, test_dir, vmap, maximages=None, size=50, stride=50, test='default'):
        self.main_dir = main_dir
        self.to_test = test_dir
        self.vmap = vmap
        self.maximages = maximages
        self.size = size
        self.stride = stride

        if test in self.test_types:
            self.test = test
        else:
            self.test = 'default'

        if not os.path.isdir(self.main_dir):
            raise ValueError

        if not os.path.isdir(self.to_test):
            raise ValueError

        if not os.path.isfile(self.vmap):
            raise ValueError

    def test_embedding(self):
        directories = get_directories(self.main_dir)
        test_images = get_dvs_images(self.to_test)

        tests = {}

        for x in test_images:
            path = os.path.join(self.to_test, x)
            tests[path] = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        others = {}

        for x in directories:
            path = os.path.join(self.main_dir, x)
            others[path] = {}

            other_images = get_dvs_images(path)

            for y in other_images:
                others[path][os.path.join(path, y)] = \
                    cv2.imread(os.path.join(self.main_dir, os.path.join(path, y)),
                               cv2.IMREAD_UNCHANGED)

        self.run_test(tests, others)

    def run_test(self, tests, others):
        f = open(self.vmap, 'r')
        vmap = []

        for i, line in enumerate(f.readlines()):
            im = img2vec.np_vec2c_vec(line[1:])
            vmap.append(im)

        f.close()

        if self.test == 'default':
            [tests, others] = preprocess_default_tests(tests, others, vmap, maximages=self.maximages)
        elif self.test == 'window':
            [tests, others] = preprocess_window_tests(tests, others, vmap, maximages=self.maximages,
                                                      size=self.size, stride=self.stride)

        open('results.txt', 'w').close()

        f = open('results.txt', 'w')

        results = {}

        for test in tests.keys():
            results[test] = {}

            for other in others.keys():
                for target in others[other].keys():
                    results[test][target] = \
                        img2vec.safe_hamming(tests[test], others[other][target])

            results[test] = sorted(results[test].items(), key=operator.itemgetter(1))

        for result in results.keys():
            output = 'Sorted results for: ' + str(result)
            f.write(output + '\n')

            print(output)

            for image in results[result]:
                output = '\t' + str(image[1]) + ' --> ' + str(image[0])
                f.write(output + '\n')

                print(output)

        f.close()

        return


parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('subdir')
parser.add_argument('vmap')
parser.add_argument('--maximages')
parser.add_argument('--size')
parser.add_argument('--stride')
parser.add_argument('--test')

args = parser.parse_args()

main = args.dir
to_test = args.subdir
v_map = args.vmap
max_im = int(args.maximages)
window_size = int(args.size)
window_stride = int(args.stride)
test_type = args.test

if window_size is None:
    window_size = 50

if window_stride is None:
    window_stride = 50

if test_type is None:
    test_type = 'default'

if test_type is not None and test_type not in EmbeddingTester.test_types:
    test_type = 'default'

tester = EmbeddingTester(main, to_test, v_map,
                         maximages=max_im, size=window_size,
                         stride=window_stride, test=test_type)

tester.test_embedding()
