from __future__ import print_function 
import struct
from numpy import empty, zeros, float32, int32, dtype, fromstring
from libc.stdio cimport *
from libc.String cimport memset
from nqe.findmax import loadvectors
from tools.types cimport *
from numpy cimport *
from tools.blas cimport saxpy, scopy, snrm2

cdef int iONE = 1
cdef float fmONE = -1.0
cdef float fONE = 1.0

cdef class kmeans:
    def __init__(self, table, assignments, threads, k):
        self.k = k
        self.embeddingcount = table.shape[0]
        self.vectorsize = table.shape[1]
        self.vectorcount = (self.embeddingcount * (self.embeddingcount - 1)) / 2
        self.threads = threads
        self.npassignment = zeros(self.vectorcount, dtype=int32)
        self.assignment = toIArray(self.npassignment)
        self.workspace = allocRP(threads)
        self.workspace2 = allocRP(threads)
        self.embedding = toRArray(table)
        self.npwordpair = empty((self.vectorcount, 2), dtype=int32)
        self.seedPairs()

        for i in range(threads):
            self.workspace[i] = allocR(self.vectorsize)
            self.workspace2[i] = allocR(self.vectorsize)

        self.centroid = allocRP(k)
        self.centroidcount = allocIntZeros(k)
        for i in range(k):
            self.centroid[i] = allocR(self.vectorsize)
            scopy(&self.vectorsize, self.vector[i], &iONE, self.centroid[i], &iONE)
        self.chooserandom = k
        print("end init")

    cdef void seedPairs(self):
        cdef int count = 0
        self.vector = allocRP(self.vectorcount)
        for i in range(self.embeddingcount-1):
            for j in range(i + 1, self.embeddingcount):
                self.vector[count] = allocR(self.vectorsize)
                scopy(&self.vectorsize, &self.embedding[i], &iONE, self.vector[count], &iONE)
                saxpy(&self.vectorsize, &fmONE, &self.embedding[j], &iONE, self.vector[count], &iONE)
                self.npwordpair[count] = (i, j)
                count += 1
        print("seeded %d", self.vectorcount)

    cdef void assignCentroids(self, int thread, int start, int end):
        cdef int v, c
        cdef float mindistance, dist
        cdef cREAL *work = self.workspace[thread]

        print("assign")
        for v in range(start, end):
            mindistance = -1
            for c in range(self.k):
                scopy(&self.vectorsize, self.centroid[c], &iONE, work, &iONE)
                saxpy(&self.vectorsize, &fmONE, self.vector[v], &iONE, work, &iONE)
                dist = snrm2(&self.vectorsize, work, &iONE)
                if mindistance == -1 or dist < mindistance:
                    mindistance = dist
                    self.assignment[v] = c

    cdef void positionCentroids(self, int thread, int start, int end):
        cdef int v, c, oldcount
        cdef float mindistance, dist
        cdef cREAL *work = self.workspace[thread]
        cdef cREAL *work2 = self.workspace2[thread]
        cdef int count
        cdef float frac

        print("position")
        for c in range(start, end):
            oldcount = self.centroidcount[c]
            self.centroidcount[c] = 0
            for v in range(self.vectorcount):
                if self.assignment[v] == c:
                    if self.centroidcount[c] == 0:
                        scopy(&self.vectorsize, self.vector[v], &iONE, work, &iONE)
                    else:
                        saxpy(&self.vectorsize, &fONE, self.vector[v], &iONE, work, &iONE)
                    self.centroidcount[c] += 1
            if oldcount == 1 and self.centroidcount[c] == 1: # teleport centroid
                scopy(&self.vectorsize, self.vector[self.chooserandom], &iONE, self.centroid[c], &iONE)
                self.chooserandom += 1
            else:
                frac = 1.0 / self.centroidcount[c]
                memset(work2, 0, self.vectorsize * 4)
                saxpy(&self.vectorsize, &frac, work, &iONE, work2, &iONE)
                scopy(&self.vectorsize, work2, &iONE, self.centroid[c], &iONE)

    cdef void p(self):
        cdef float dist
        cdef cREAL *work = self.workspace[0]
        for v in range(self.vectorcount):
            scopy( &self.vectorsize, self.centroid[self.assignment[v]], & iONE, work, & iONE)
            saxpy( &self.vectorsize, & fmONE, self.vector[v], & iONE, work, & iONE)
            dist = snrm2( &self.vectorsize, work, & iONE)
            printf("%3d %f ", self.assignment[v], dist)
        printf("\n")

    def listC(self, vec2word, assignments, c):
        for i in range(self.vectorcount):
            if assignments[i] == c:
                yield vec2word[self.wordpair[i][0]] + "-" + vec2word[self.wordpair[i][1]]

    def printcluster(self, assignments, vec2word):
        for c in range(self.k):
            for w in self.listC(vec2word, assignments, c):
                print(w, c)

    cdef void p2(self, int c):
        for v in range(self.vectorsize):
            printf("%f ", self.centroid[c][v])
        printf("\n")

def learn(fname, iterations):
    table, word2vec, vec2word = loadvectors(fname)
    assignments = zeros(len(word2vec), dtype=int32)
    print("loaded %d", len(word2vec))
    k = kmeans(table, assignments, 1, 100)
    for i in range(iterations):
        k.assignCentroids(0, 0, k.vectorcount)
        k.positionCentroids(0, 0, k.k)
        k.p()
    k.printcluster(assignments, vec2word)



