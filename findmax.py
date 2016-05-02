from nqe.findmax import *
import sys
import numpy

if __name__ == "__main__":
    vectorfile = "data/part1"
    wordfile = "data/questions-words.txt"
    table, word2vec, vec2word = loadvectors(vectorfile)
    words = loadAnalogy(wordfile, word2vec)
    print(words)
    words = numpy.array([ 1, 2, 5, 6, 8, 10 ])
    a = findmax(words, table)
    print(a)


