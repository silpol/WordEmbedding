'''
Use this class to encode high-dimensional NGrams to short (low-dimensional) word vectors

import CGcA
CGcA.run()
'''

# Python modules
import cPickle
import numpy
import theano
import theano.tensor as T
import time

import sys
sys.path.append("..")

# Codegen modules
from offline_paths import OfflinePaths
import textUtil


"""This tutorial introduces Contractive auto-encoders (cA) using Theano.

 They are based on auto-encoders as the ones used in Bengio et
 al. 2007.  An autoencoder takes an input x and first maps it to a
 hidden representation y = f_{\theta}(x) = s(Wx+b), parameterized by
 \theta={W,b}. The resulting latent representation y is then mapped
 back to a "reconstructed" vector z \in [0,1]^d in input space z =
 g_{\theta'}(y) = s(W'y + b').  The weight matrix W' can optionally be
 constrained such that W' = W^T, in which case the autoencoder is said
 to have tied weights. The network is trained such that to minimize
 the reconstruction error (the error between x and z).  Adding the
 squared Frobenius norm of the Jacobian of the hidden mapping h with
 respect to the visible units yields the contractive auto-encoder:

      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]
      + \| \frac{\partial h(x)}{\partial x} \|^2

 References :
   - S. Rifai, P. Vincent, X. Muller, X. Glorot, Y. Bengio: Contractive
   Auto-Encoders: Explicit Invariance During Feature Extraction, ICML-11

   - S. Rifai, X. Muller, X. Glorot, G. Mesnil, Y. Bengio, and Pascal
     Vincent. Learning invariant features through local space
     contraction. Technical Report 1360, Universite de Montreal

   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

class cA(object):
    """ Contractive Auto-Encoder class (cA)

    The contractive autoencoder tries to reconstruct the input with an
    additional constraint on the latent space. With the objective of
    obtaining a robust representation of the input space, we
    regularize the L2 norm(Froebenius) of the jacobian of the hidden
    representation with respect to the input. Please refer to Rifai et
    al.,2011 for more details.

    If x is the input then equation (1) computes the projection of the
    input into the latent space h. Equation (2) computes the jacobian
    of h with respect to x.  Equation (3) computes the reconstruction
    of the input, while equation (4) computes the reconstruction
    error and the added regularization term from Eq.(2).

    .. math::

        h_i = s(W_i x + b_i)                                             (1)

        J_i = h_i (1 - h_i) * W_i                                        (2)

        x' = s(W' h  + b')                                               (3)

        L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
             + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

    """

    def __init__(self, numpy_rng = numpy.random.RandomState(123), input=None, n_visible=3, n_hidden=2,
                 n_batchsize=1, W=None, bhid=None, bvis=None):
        """Initialize the cA class by specifying the number of visible units
        (the dimension d of the input), the number of hidden units (the
        dimension d' of the latent or hidden space) and the contraction level.
        The constructor also receives symbolic variables for the input, weights
        and bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given
                     one is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone cA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type n_batchsize int
        :param n_batchsize: number of examples per batch

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                )
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the cA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L_rec = - T.sum(self.x * T.log(z) +
                             (1 - self.x) * T.log(1 - z),
                             axis=0)

        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = T.sum(J ** 2) / self.n_batchsize

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def shared_dataset(data, borrow=True):
        shared =theano.shared(numpy.asarray(data),
                                 borrow=borrow)
        return T.cast(shared, 'int32')

#CodeGen code
class CGcA:
    def __init__(self, inputPath, outputPath):
        self.inputPath = inputPath
        self.outputPath = outputPath


        self.learningRate = .1
        self.trainingEpochs = []
        self.contractionLevel = .1

        self.epochErrors = []
        self.networkParams = None

        self.vocabulary = []
        self.vocabularySize = 0
        self.trainSet = None

        self.encodeDimension = 32
        self.batchSize = 300
        self.x = None
        self.ca = None

        self.shortWordVectors = []

    def __getstate__(self):
        #Return state values to be pickled.
        d = dict(self.__dict__)
        del d['vocabulary']
        del d['trainSet']
        return d

    def loadTrainSet(self):
        NGrams = cPickle.load(open(self.inputPath, "rb")).matNGrams
        self.vocabularySize = len(NGrams)
        self.vocabulary = []
        for i in range(self.vocabularySize):
            self.vocabulary.append(textUtil.getSparseNGram(NGrams[i], self.vocabularySize))
        self.trainSet = theano.shared(numpy.asarray(self.vocabulary, dtype = theano.config.floatX), borrow = True)

    def prepareCA(self):
        self.x = T.matrix('x')  # the data is presented as sparse word vectors array

        if self.networkParams is None:
            self.ca = cA(input=self.x, n_visible=self.vocabularySize, n_hidden=self.encodeDimension, n_batchsize=self.batchSize)
        else:
            self.ca = cA(input=self.x, n_visible=self.vocabularySize, n_hidden=self.encodeDimension, n_batchsize=self.batchSize,
                           W=self.networkParams[0], bhid=self.networkParams[1], bvis=self.networkParams[2])

    def trainCA(self, noEpochs = 1):
        cost, updates = self.ca.get_cost_updates(contraction_level=self.contractionLevel, learning_rate=self.learningRate)

        index = T.lscalar()    # index to a [mini]batch
        train_ca = theano.function(
            [index],
            [T.mean(self.ca.L_rec), self.ca.L_jacob],
            updates=updates,
            givens={
                self.x: self.trainSet[index * self.batchSize: (index + 1) * self.batchSize]
            }
        )

        # go through training epochs
        noBatches = self.vocabularySize / self.batchSize
        print "batches:", noBatches
        startEpoch = 0
        for noEp in self.trainingEpochs:
            startEpoch += noEp
        self.trainingEpochs.append(noEpochs)
        for epoch in xrange(noEpochs):
            start_time = time.clock()
            # go through trainng set
            c = []
            for batch_index in xrange(noBatches):
                    c.append(train_ca(batch_index))
                    print 'Batch ', batch_index

            c_array = numpy.vstack(c)
            self.epochErrors.append(numpy.mean(c_array[0]))
            
            print 'Training epoch %d, reconstruction cost %d, run-time %d minutes ' \
            % ((epoch + startEpoch), numpy.mean(c_array[0]), (time.clock() - start_time) / 60)
        self.networkParams = self.ca.params
            

    def getShortWordVectors(self):
        self.shortWordVectors = []
        for word in self.vocabulary:
            tWord = theano.shared(numpy.asarray(word, dtype=theano.config.floatX), borrow=True)
            self.shortWordVectors.append(self.ca.get_hidden_values(tWord).eval())
            
def run(fInput = None, fOutput = None, noEpochs = 1):
    networkState = None
    if fOutput:
        networkState = cPickle.load(open(fOutput, "rb"))
        networkState.ouputPath = fOutput
        printErrors(networkState)
    else:
        networkState = CGcA(OfflinePaths.cgparse_ngrams, OfflinePaths.cgca_state)
        
    if fInput:
        networkState.inputPath = fInput
        if fOutput:
            viewSimilarities(networkState, "prostate")

    print "loading train set, from ", networkState.inputPath
    networkState.loadTrainSet()

    print "preparing network"
    networkState.prepareCA()

    print "training network"
    networkState.trainCA(noEpochs)

    print "computing contracted word vectors (hidden layer activation)"
    networkState.getShortWordVectors()

    print "dumping network state (excluding training set) to ", networkState.outputPath
    cPickle.dump(networkState, open(networkState.outputPath, "wb"))
    
    printErrors(networkState)
    viewSimilarities(networkState, "prostate")

def printErrors(networkState):
    print "Errors:"
    for err in networkState.epochErrors:
        print err
        
def viewSimilarities(networkState, sWord):
    parser = cPickle.load(open(networkState.inputPath,"rb"))
    layers = networkState.shortWordVectors
    noRecords = 50
    
    print "Word :",sWord
    similarities = textUtil.getCosineSimilarities(parser.dictWords[sWord].id, parser.matNGrams)
    print "Original, cosine: ", textUtil.printTupleList(similarities, parser.listWords)[:noRecords] 
    #similarities = textUtil.getEuclideanSimilarities(parser.dictWords[sWord].id, parser.matNGrams)
    #print "Original, euclidean: ", textUtil.printTupleList(similarities, parser.listWords)[:noRecords]

    listLayers=[i.tolist() for i in layers]
    similarities= textUtil.getCosineSimilarities(parser.dictWords[sWord].id, listLayers)
    print "Contracted, cosine: ", textUtil.printTupleList(similarities, parser.listWords)[:noRecords]
    #similarities= textUtil.getEuclideanSimilarities(parser.dictWords[sWord].id,listLayers)
    #print "Contracted, euclidean: ", textUtil.printTupleList(similarities,parser.listWords)[:noRecords]
    
#run(fInput = r'C:\work\codegen\codegen\test_data\TextParsing\CGParse_NGrams.p',fOutput = r'C:\work\codegen\codegen\test_data\TextParsing\CGcA_state.p')
#run(noEpochs=30)
