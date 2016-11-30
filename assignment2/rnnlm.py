from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid , make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        self.sparams.L = random_weight_matrix(*self.sparams.L.shape)
        self.sparams.U = random_weight_matrix(*self.params.U.shape)
        
        self.params.H = random_weight_matrix(*self.params.H.shape)

        self.bptt = bptt
        self.alpha = alpha

    def _acc_grads(self, xs, ys):


        # Expect xs as list of indices
        ns = len(xs)

        #print 'size of window ',ns

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        
        #print self.params.U.shape
        #print self.params.H.shape
        #print self.sparams.L.shape
        #bre
        for t in range(0,ns):
            hs[t] = sigmoid(dot(hs[t-1],self.params.H)+self.sparams.L[xs[t],:])
            ps[t] = softmax(dot(self.params.U,hs[t]))

        #print hs
        temp_ys = []
        for i in range(0,ns):
            temp_ys.append(make_onehot(ys[i],self.vdim))

        temp_ys = matrix(temp_ys)
        #print hs
        #print ps
        #print ps.shape
        #print temp_ys.shape
        delta2 = ps - temp_ys

        #print self.grads.U

        ##
        # Backward propagation through time

        for t in range(0,ns)[::-1]:
            self.grads.U += outer(delta2[t],hs[t])
            #print self.grads.U
            delta1 = multiply(dot(delta2[t],self.params.U),hs[t]*(1-hs[t]))
            for step in range(max(0,t-self.bptt),t+1)[::-1]:
                self.grads.H += dot(delta1,hs[step-1])
                self.sgrads.L[xs[step]] = delta1
                delta1 = multiply(dot(delta1,self.params.H),hs[step-1]*(1-hs[step-1]))


        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):

        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):

        J = 0
        #### YOUR CODE HERE ####
        
        ns = len(xs)
        hs = zeros((ns+1, self.hdim))
        ps = zeros((ns, self.vdim))

        # Forward propagation
        for t in range(0,ns):
            hs[t] = sigmoid(dot(hs[t-1],self.params.H)+self.sparams.L[xs[t],:])
            ps[t] = softmax(dot(self.params.U,hs[t]))
            J += -log(ps[t][ys[t]])
        
        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):

        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):

        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):

        J = 0 # total loss
        ys = [init] # emitted sequence

        #### YOUR CODE HERE ####


        #### YOUR CODE HERE ####
        return ys, J



class ExtraCreditRNNLM(RNNLM):

    def __init__(self, *args, **kwargs):
        #### YOUR CODE HERE ####
        raise NotImplementedError("__init__() not yet implemented.")
        #### END YOUR CODE ####

    def _acc_grads(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("_acc_grads() not yet implemented.")
        #### END YOUR CODE ####

    def compute_seq_loss(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("compute_seq_loss() not yet implemented.")
        #### END YOUR CODE ####

    def generate_sequence(self, init, end, maxlen=100):
        #### YOUR CODE HERE ####
        raise NotImplementedError("generate_sequence() not yet implemented.")
        #### END YOUR CODE ####
