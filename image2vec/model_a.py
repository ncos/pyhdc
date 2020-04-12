import numpy as np
import os, sys, shutil, signal, glob, time
from image2vec import *


class Memory_a:
    def __init__(self):
        self.basis_vectors = []
        self.masked_vectors = []
        self.vcount = 0
        self.m = pyhdc.LBV()

    def add(self, v):
        self.vcount += 1
        x = pyhdc.LBV() 
        #x.rand()
        self.basis_vectors.append(x)

        masked_v = pyhdc.LBV()
        masked_v.xor(v)
        masked_v.xor(x)
        self.masked_vectors.append(masked_v)

    def build(self, to_adjust=-1):
        #print ("\tmemory vectors:", self.vcount)
        #if (to_adjust > 0): print ("\t\tadjusting to:", to_adjust)

        self.m.xor(self.m)
        if (self.vcount == 0):
            self.m.rand()
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = pyhdc.LBV()
            x.rand()
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors
        if (len(csum_vectors) == 1):
            self.m.xor(csum_vectors[0])
            self.masked_vectors = []
            return

        th = self.vcount // 2
        for i in range(pyhdc.get_vector_width()):
            cnt = 0
            for v in csum_vectors:
                if (v.get_bit(i)):
                    cnt += 1
            if (cnt >= th):
                self.m.flip(i)

        #print ("Outlier check...")
        #for v in self.masked_vectors:
        #    score, id_ = self.find(v)
        #    print ("\t", score, id_)
        #print ("\n")
        self.masked_vectors = []

    def find(self, v):
        mem_test = pyhdc.LBV()
        mem_test.xor(self.m)
        mem_test.xor(v)

        min_score = pyhdc.get_vector_width()
        min_id = -1

        for i, b in enumerate(self.basis_vectors):
            tmp = pyhdc.LBV()
            tmp.xor(mem_test)
            tmp.xor(b)
            score = tmp.count()

            if (min_score > score):
                min_score = score
                min_id = i
                break
        return min_score, min_id


class Model_a:
    def __init__(self, m):
        self.cl_mapper = m
        self.bins = {}
        self.infer_db = []

    def add(self, vec_image, val):
        classes = self.cl_mapper.get_class(val)
        #print ("\tassigning classes:", classes)
        for cl in classes:
            if cl not in self.bins.keys():
                self.bins[cl] = Memory_a()
            self.bins[cl].add(vec_image.vec)
    
    def build(self):
        print ("Bulding the model:", len(self.bins.keys()), "clusters")
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        #print ("Adjusting memory bins to:", to_adjust)
        for i, cl in enumerate(sorted(self.bins.keys())):
            #print ("Building memory for cluster", cl, "(", i, "/", len(self.bins.keys()), ")")
            print ("\tcluster", cl, ":\t", self.bins[cl].vcount)
            self.bins[cl].build(-1)

    def infer(self, vec_image):
        clusters = []
        scores = []

        for i, cl in enumerate(sorted(self.bins.keys())):
            #print ("\tLooking for a vector in memory", cl, "(", i, "/", len(self.bins.keys()), ")")
            if (self.bins[cl].vcount <= 2):
                continue
            
            score, id_ = self.bins[cl].find(vec_image.vec)
            #print ("\tScore:", score, "\tbasis id:", id_)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        scores -= np.min(scores)
        scores /= np.max(scores)
        scores = 1 - scores

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))
        #print ("\tresults:", zip(scores, clusters))
        #print ("\tcluster:", result[-1][1])

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]


