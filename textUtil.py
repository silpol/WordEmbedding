import math
from whoosh.analysis import *

genoRef = re.compile(r'((gs[1-9][0-9]*)|((rs|i)[0-9]+(\([A-Z-;]+\))))')
alphaPattern = re.compile(r'[\W_]+')
standAloneNumberPattern = re.compile(r'\b[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\b')


def sentence2words(sentence):
    return unigrams(sentence)


def unigrams(text):
    rez = []
    # lowercase, remove stopwords, replace rs1234(A;B) with re1234AB,
    # replace numbers with generic placeholder
    # whoosh porter stemming is rubbish do not use
    text = genoRef.sub(lambda matchobj: re.sub(alphaPattern, '', matchobj.group(0)), text)
    tokenizer = RegexTokenizer() | LowercaseFilter() | StopFilter()

    for token in tokenizer(unicode(text, errors='replace')):
        if standAloneNumberPattern.match(token.text) is not None:
            rez.append('floatvalue')
        elif len(token.text) > 2:
            rez.append(token.text)
    return rez


def bigrams(token_list, sep=' '):
    last = None
    rez = []
    for token in token_list:
        if last is not None:
            rez.append(last + sep + token)
        last = token
    return rez


def getTfIdfForDocs(documents, includeBigrams=False, docIdFunction=None, textFunction=None):
    docs = {}
    if docIdFunction is None:
        docIdFunction = lambda x: x
    if textFunction is None:
        textFunction = lambda x: x
    # build a dictionary of tokens to number of documents containing each token
    idf = {}
    for doc in documents:
        # extract ngrams
        tokens = unigrams(textFunction(doc))
        # may want to consider bigrams e.g. prostate cancer
        if includeBigrams:
            tokens.extend(bigrams(tokens))
        docs[docIdFunction(doc)] = {'freq': {}, 'tf': {}, 'tf-idf': {}}
        for token in tokens:
            # The frequency computed for each document
            docs[docIdFunction(doc)]['freq'][token] = tokens.count(token)
            # The term-frequency (Normalized Frequency)
            docs[docIdFunction(doc)]['tf'][token] = docs[docIdFunction(doc)]['freq'][token] / float(len(tokens))
            idf[token] = 1.0

    for doc in docs:
        for token in docs[doc]['tf']:
            idf[token] += 1

    docCount = len(docs)
    # now transform the token to doc counts to IDF
    for token in idf:
        idf[token] = math.log(docCount / idf[token])
    for doc in docs:
        for token in docs[doc]['tf']:
            # The tf-idf
            docs[doc]['tf-idf'][token] = docs[doc]['tf'][token] * idf[token]
    # flatten the vocab
    return docs, idf


def expandDocVector(tfIdfDict, vocabulary, docId):
    return [(tfIdfDict[docId]['tf-idf'][word] if word in tfIdfDict[docId]['tf-idf'] else float(0)) for word in vocabulary]


def expandAllDocVectors(tfIdfDict, vocabulary):
    return [expandDocVector(tfIdfDict, vocabulary, docId) for docId in tfIdfDict]


'''
Normalize a whole matrix
data: a dense matrix [{}]
return: the normalized dense matrix [{}]
'''
def getNormalMatrix(data):
    vocabSize = len(data)
    resNormed = [{} for i in range(vocabSize)]
    norm = getMatrixNorm(data)

    for i in range(vocabSize):
        dictEmbedding = data[i]
        dictNormedEmbedding = resNormed[i]
        for key in dictEmbedding:
            dictNormedEmbedding[key] = dictEmbedding[key]/norm
    return resNormed
'''
input: matrix [[]] or dense matrix [{}]
output: float
'''
def getMatrixNorm(data):
    rowCount = len(data)
    rez = 0

    for i in range(rowCount):
        cVector = data[i]
        if type(cVector) == type([]):
            for val in cVector:
                rez += math.pow(val, 2)
        elif type(cVector) == type({}):
            for val in cVector.values():
                rez += math.pow(val, 2)
        else:
            raise AssertionError("matrix rows must be list or dictionary")
    return math.sqrt(rez)

'''
Get a list of sinonimes for a vector in a vector array
wordIdx: the index of the target word in data
data: matrix [[]] or dense matrix [{}], one word representation per row
'''
def getCosineSimilarities(wordIdx, data):
    dictRez = {}
    word = data[wordIdx]
    norms = getRowNorms(data)

    for i in range(len(data)):
        pair = data[i]
        scalarProd = 0
        if type(word) == type([]):
            for j in range(len(word)):
                scalarProd += word[j] * pair[j]
        elif type(word) == type({}):
            for key in word:
                if key in pair:
                    scalarProd += word[key] * pair[key]
        else:
            raise AssertionError("matrix rows must be list or dictionary")
        if scalarProd != 0:
            dictRez[i] =scalarProd / (norms[wordIdx] * norms[i])
    return sorted(dictRez.items(), key=lambda x: x[1], reverse=True)
'''
Get a list of sinonimes for a vector in a vector array
wordIdx: the index of the target word in data
data: matrix [[]] or dense matrix [{}], one word representation per row
'''
def getEuclideanSimilarities(wordIdx, data):
    dictRez = {}
    word = data[wordIdx]

    for i in range(len(data)):
        pair = data[i]
        euclDist = 0
        if type(word)==type([]): #sparse vector
            for j in range(len(word)):
                euclDist += math.pow(word[j] - pair[j], 2)
        elif type(word)==type({}): #dense vector
            for key in word:
                if key in pair:
                    euclDist += math.pow(word[key] - pair[key], 2)
                else:
                    euclDist += math.pow(word[key], 2)
            for key in pair:
                if key not in word:
                    euclDist += math.pow(pair[key], 2)
        else:
            raise AssertionError("matrix rows must be list or dictionary")
        dictRez[i]=math.sqrt(euclDist)
    return sorted(dictRez.items(), key=lambda x:x[1], reverse=False)
'''
Get the norms list for a vector array
input: matrix [[]] or dense matrix [{}]
output: a vector, one element for each input row
'''
def getRowNorms(data):
    rowCount = len(data)
    listRez = [0 for i in range(rowCount)]

    for i in range(rowCount):
        cVector = data[i]
        cNorm = 0
        if type(cVector) == type([]):
            for val in cVector:
                cNorm += math.pow(val, 2)
        elif type(cVector) == type({}):
            for val in cVector.values():
                cNorm += math.pow(val, 2)
        else:
            raise AssertionError("matrix rows must be list or dictionary")
        listRez[i] = math.sqrt(cNorm)
    return listRez

'''
This method 'translates' a tuple such as (1034,0.63245) to ('ibuprofen',0.63245), based on the listWords indexes
listTuple: [(wordIndex,distance)]
listWords: [strWord]
'''
def translateTupleList(listTuple, listWords):
    listRez = []
    for tup in listTuple:
        listRez.append((listWords[tup[0]], tup[1]))
    return listRez
def printTupleList(listTuple, listWords):
    tuples = translateTupleList(listTuple, listWords)
    listRez = []
    for tup in tuples:
        listRez.append(tup[0])
    return listRez

'''
Return a high-dimensional word vector based on the dense dictionary (wordIndex,wordCount)
word:  {}
vocabSize: int
output: []
'''
def getSparseNGram(word,vocabSize):
    rez = [0 for i in range(vocabSize)]
    for key in word:
        rez[key] = word[key]
    return rez
