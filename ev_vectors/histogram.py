#!/usr/bin/python3

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2


class Pattern:
    def __init__(self, values, template):
        self.values = values
        self.template = template
        self.count = 0
        self.hash = self.hashfunc(self.values, self.template.id)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash

    def hashfunc(self, vec, id_):
        h = id_
        for i, v in enumerate(vec):
            h += int(np.power(256, i + 1)) * v
        return h

    def left(self):
        if (self.template.level == 1):
            print ("Should not be called!")
            return self.__hash__()

        vals = [self.values[i] for i in self.template.left_hash_sequence]
        return self.hashfunc(vals, self.template.left_id)

    def right(self):
        if (self.template.level == 1):
            print ("Should not be called!")
            return self.__hash__()

        vals = [self.values[i] for i in self.template.right_hash_sequence]
        return self.hashfunc(vals, self.template.right_id)

    def inc(self):
        self.count += 1

    def __repr__(self):
        return str(self.values) + " <-> " + str(self.__hash__())


class PatternTemplate:
    def __init__(self, mask, id_, left=None, right=None):
        self.mask = mask
        self.level = len(self.mask)
        self.id = id_

        if (left is None and right is None and self.level == 1):
            return

        self.left_id  = left[0]
        self.right_id = right[0]

        self.left_hash_sequence  = left[1:]
        self.right_hash_sequence = right[1:]


    def extract(self, image, i, j):
        values = []
        for p in self.mask:
            val = int(image[i + p[0], j + p[1]])
            if (val < 1):
                return None
            values.append(val)
        return Pattern(values, self)


class PatternHistogram:
    def __init__(self, img):
        # Make sure it is the grayscale image
        if (len(img.shape) > 2 and img.shape[2] == 3):
            self.image = img[:,:,1]
        elif (len(img.shape) == 2):
            self.image = img
        else:
            print ("Unsupported image size: ", self.image.shape)

        print ("Image shape:", self.image.shape, "Type:", self.image.dtype)

        self.histograms = []
        self.pattern_templates = []

        self.generate_patterns()
        self.build_histogram()

    # Patterns are masks in a 3x3 region
    def generate_patterns(self):
        #                                             [mask]               id  left/right = (child_id, elem. mapping)      
        self.pattern_templates.append(PatternTemplate([(1,1)],             0))
        self.pattern_templates.append(PatternTemplate([(2,1),(1,1)],       1, left=(0, 0),    right=(0, 1)))
        self.pattern_templates.append(PatternTemplate([(1,1),(1,2)],       2, left=(0, 0),    right=(0, 1)))
        #self.pattern_templates.append(PatternTemplate([(2,1),(1,1),(0,1)], 3, left=(1, 0, 1), right=(1, 1, 2)))
        #self.pattern_templates.append(PatternTemplate([(2,1),(1,1),(1,2)], 4, left=(1, 0, 1), right=(2, 1, 2)))
        #self.pattern_templates.append(PatternTemplate([(1,0),(1,1),(1,2)], 5, left=(2, 0, 1), right=(2, 1, 2)))
        #self.pattern_templates.append(PatternTemplate([(1,0),(1,1),(0,1)], 6, left=(2, 0, 1), right=(1, 1, 2)))
        #self.pattern_templates.append(PatternTemplate([(0,1),(1,1),(1,2)], 7, left=(1, 1, 0), right=(2, 1, 2)))
        #self.pattern_templates.append(PatternTemplate([(2,1),(1,1),(1,0)], 8, left=(1, 0, 1), right=(2, 2, 1)))

        for pattern in self.pattern_templates:
            self.histograms.append({})

    def build_histogram(self):
        for i in range(0, self.image.shape[0] - 2):
            for j in range(0, self.image.shape[1] - 2):
                for pattern_template in self.pattern_templates:
                    pattern = pattern_template.extract(self.image, i, j)
                    if (pattern is None):
                        continue
                   
                    if pattern not in self.histograms[pattern_template.id].keys():
                        self.histograms[pattern_template.id][pattern] = pattern
                    self.histograms[pattern_template.id][pattern].inc()


    def print_stats(self):
        for i, hist in enumerate(self.histograms):
            print ("Pattern", i, "has", len(hist), "elements")
            mincnt = int(np.power(256, 3))
            maxcnt = 0
            for pattern in hist:
                cnt = pattern.count
                if (cnt < mincnt): mincnt = cnt
                if (cnt > maxcnt): maxcnt = cnt
            print ("\t\tmin/max counts:", mincnt, maxcnt)

    
    def generate_gtaph(self):
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default="",
                        required=False)
    parser.add_argument('--image',
                        type=str,
                        default="",
                        required=False)
    args = parser.parse_args()

    print ("Opening", args.base_dir)

    # load dataset
    image_filenames = []
    gt_lines = []
    if (args.base_dir != ""):
        with open(os.path.join(args.base_dir, 'cam_vels_local_frame.txt')) as f:
            gt_lines = f.readlines()
        
        for line in gt_lines:
            fname = gt_lines.split(' ')[0]
            image_filenames.append(os.path.join(args.base_dir, 'slices', fname))

    if (args.image != ""):
        image_filenames.append(args.image)

    if (len(image_filenames) == 0):
        print ("No images to process!")
        exit(-1)

    pattern_histograms = [PatternHistogram(cv2.imread(img_name, cv2.IMREAD_UNCHANGED)) for img_name in image_filenames]
    
    for ph in pattern_histograms:
        ph.print_stats()
