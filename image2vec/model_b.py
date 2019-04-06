import numpy as np
import os, sys, shutil, signal, glob, time
from image2vec import *


class Memory_b:
    def __init__(self):
        self.basis_vectors = {}
        self.masked_vectors = {}
        self.vcount = 0
        self.m = pyhdc.LBV()

    def add(self, v, cl):
        if (cl not in self.basis_vectors.keys()):
            self.vcount += 1
            x = pyhdc.LBV() 
            for i in range(abs(cl) + 5):
                x.rand()
            self.basis_vectors[cl] = x

        masked_v = pyhdc.LBV()
        masked_v.xor(v)
        masked_v.xor(self.basis_vectors[cl])
        
        if (cl not in self.masked_vectors.keys()):
            self.masked_vectors[cl] = []
        self.masked_vectors[cl].append(masked_v)

    def build(self):
        print ("\tmemory vectors:", self.vcount)
        #if (to_adjust > 0): print ("\t\tadjusting to:", to_adjust)

        self.m.xor(self.m)
        if (self.vcount == 0):
            self.m.rand()
            return

        cluster_vectors = []
        for cl in sorted(self.basis_vectors.keys()):
            print ("\tcluster", cl, "has", len(self.masked_vectors[cl]), "vectors")
            
            cluster_vectors.append(self.csum(self.masked_vectors[cl]))
            #cluster_vectors += self.masked_vectors[cl]

        self.m.xor(self.csum(cluster_vectors))
        self.masked_vectors = {}

    def csum(self, vectors):
        res = pyhdc.LBV()
        if (len(vectors) == 0):
            return res
        if (len(vectors) == 1):
            return vectors[0]

        th = len(vectors) // 2 
        for i in range(pyhdc.get_vector_width()):
            cnt = 0
            for v in vectors:
                if (v.get_bit(i)):
                    cnt += 1
            if (cnt >= th):
                res.flip(i)
        return res

    def find(self, v):
        mem_test = pyhdc.LBV()
        mem_test.xor(self.m)
        mem_test.xor(v)

        min_score = pyhdc.get_vector_width()
        min_id = -1
        
        clusters = []
        scores = []
        for i, cl in enumerate(sorted(self.basis_vectors.keys())):
            tmp = pyhdc.LBV()
            tmp.xor(mem_test)
            tmp.xor(self.basis_vectors[cl])
            score = tmp.count()

            clusters.append(cl)
            scores.append(score)

            if (min_score > score):
                min_score = score
                min_id = i
        return min_score, min_id, clusters, scores


class Model_b:
    def __init__(self, m):
        self.cl_mapper = m
        self.memory = Memory_b()
        self.infer_db = []

    def add(self, vec_image, val):
        classes = self.cl_mapper.get_class(val)
        #print ("\tassigning classes:", classes)
        for cl in classes:
            self.memory.add(vec_image.vec, cl)
     
    def build(self):
        self.memory.build()

    def infer(self, vec_image):
        clusters = []
        scores = []

        min_score, min_id, clusters, scores = self.memory.find(vec_image.vec)

        scores = np.array(scores, dtype=np.float)
        scores -= np.min(scores)
        scores /= np.max(scores)
        scores = 1 - scores

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))
        #print ("\tresults:", zip(scores, clusters))
        #print ("\tcluster:", result[-1][1])

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]


