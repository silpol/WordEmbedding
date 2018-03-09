'''
Use this class to get high-dimensional NGrams from a CodeGen thesaurus

import CGParse
CGParse.getNGrams()
'''

import sys
sys.path.append("..")

from offline_paths import OfflinePaths
from paths import BackendPaths
import textUtil
import cPickle


class CGWord:
    def __init__(self, word, id):
        self.word = word
        self.id = id        # this is the index in listWords
        self.count = 1
    def __repr__(self):
        return repr((self.word, self.id, self.count))

class CGParse:
    def __init__(self):
        self.dictWords = {} # all vocabulary words ex ("ibuprofen":CGWord("ibuprofen",id)); dictWords and listWords always have the same length
        self.listWords = [] # all vocabulary words; if listWords[x]=="ibuprofen" then dictWords["ibuprofen"].id == x
        self.matSentences = [[]] # all sentences, written as index arrays; indexes  match listWords
        self.matNGrams = [{}] # all embedings as dense arrays ex {neighbourWordIndex:count};total size and all indexes match listWords
        self.windowSize = 0 # window size used for computing matNGrams

    def loadCgData(self, fPath):
        rezSentences = []
        codegen = cPickle.load(open(fPath, "rb"))

        for k in codegen.genotypes:
            genotype = codegen.genotypes[k]

            rezSentences.append(genotype.text)
            rezSentences.append(genotype.snp.text)

            for paper in genotype.papers:
                rezSentences.append(paper.get_title())
                rezSentences.append(paper.summary)

            for paper in genotype.snp.papers:
                rezSentences.append(paper.get_title())
                rezSentences.append(paper.summary)

        for k in codegen.genosets:
            genoset = codegen.genosets[k]

            rezSentences.append(genoset.summary)
            rezSentences.append(genoset.text)

            for paper in genoset.papers:
                rezSentences.append(paper.get_title())
                rezSentences.append(paper.summary)

        return rezSentences

    def getWords(self, listRawSentences):
        self.dictWords = {}
        self.matSentences = []

        for sentence in listRawSentences:
            sentence = sentence.strip().encode('utf-8')
            if not sentence:
                continue

            listWordIndexes = []
            words = textUtil.sentence2words(sentence)
            for word in words:
                if word in self.dictWords:
                    self.dictWords[word].count += 1
                else:
                    self.dictWords[word] = CGWord(word, len(self.dictWords))
                    self.listWords.append(word)
                listWordIndexes.append(self.dictWords[word].id)
            self.matSentences.append(listWordIndexes)
        assert len(self.listWords) == len(self.dictWords)

    def getNGrams(self, windowSize):
        self.windowSize = windowSize
        iWindowRange = range(-1*(windowSize-1), windowSize)
        self.matNGrams = [{} for i in range(len(self.listWords))]

        for sentence in self.matSentences:
            sentenceLen = len(sentence)
            for i in range(sentenceLen):
                cWord = sentence[i]
                cWordEmbedding = self.matNGrams[cWord]
                for j in iWindowRange:
                    cPairIdx = i+j
                    if cPairIdx < 0 or cPairIdx >= sentenceLen or cPairIdx == i:
                        continue
                    cPair = sentence[cPairIdx]
                    if cPair == cWord:
                        continue
                    if cPair in cWordEmbedding.keys():
                        cWordEmbedding[cPair] += 1
                    else:
                        cWordEmbedding[cPair] = 1

def getNGrams():
    print "loading data from ", BackendPaths.codegenRuntime
    parser = CGParse()
    sentences = parser.loadCgData(BackendPaths.codegenRuntime)
    print "tokenizing for words"
    parser.getWords(sentences)
    print "Vocabulary size:", len(parser.listWords)

    windowSize = 7
    print "computing NGrams, ", windowSize, " words-wide windows"
    parser.getNGrams(windowSize)
    print "normalizing vocabulary vectors"
    parser.matNGrams = textUtil.getNormalMatrix(parser.matNGrams)

    outPath = OfflinePaths.cgparse_ngrams
    print "dumping data to ", outPath
    cPickle.dump(parser, open(outPath, 'wb'))
#getNGrams()