from nqe.findmax import *
import sys
import numpy

from nqe.kmeans import learn

if __name__ == "__main__":
    vectorfile = "data/selectedwords"
    learn(vectorfile, 20)


