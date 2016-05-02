from nqe.findmax import *
import sys
import numpy

if __name__ == "__main__":
    vectorfile = "data/GoogleNews-vectors-negative300.bin"
    wordfile = "data/questions-words.txt"
    words = loadAnalogyTerms(wordfile)
    print(words)
    table, word2vec, vec2word = loadvectors(vectorfile, filter=words)
    print(word2vec)
    save_binary("data/selectedwords", word2vec, table)
    print("done")


