from __future__ import print_function
import struct
from numpy import zeros, float32, int32, dtype, fromstring

from nqe.findmax import loadvectors
from tools.types cimport *
from numpy cimport *
from tools.blas cimport saxpy, scopy, snrm2

cdef class kmeans:
    cdef int k
    cdef int chooserandom
    cdef int vectorsize
    cdef int vectorcount
    cdef int embeddingcount
    cdef cREAL **centroid
    cdef cINT *centroidcount
    cdef cREAL *embedding
    cdef cREAL **vector
    cdef ndarray npassignment
    cdef cINT *assignment
    cdef ndarray npwordpair
    cdef int threads
    cdef cREAL **workspace
    cdef cREAL **workspace2

    cdef void seedPairs(self)

    cdef void assignCentroids(self, int thread, int start, int end)

    cdef void positionCentroids(self, int thread, int start, int end)

    cdef void p(self)
    cdef void p2(self, int)






