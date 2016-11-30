from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
import data_utils.utils as du


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        self.param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, self.param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        # any other initialization you need
 
        self.wv = wv
        #print self.sgrads

        #### END YOUR CODE ####

    def sigmoid(self,x):
        x = 1.0/(1+exp(-x))
        return x

    def tan_grad(self,f):
        f = 1-multiply(f,f)
        return f


    def softmax(self,x):
        x = exp(x).transpose() / sum(exp(x),axis=1).transpose()
        return x

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """

        print self.sgrads
        
        #### YOUR CODE HERE ####
        for i in self.param_dims.keys():
            dim = matrix(self.grads[i]).shape
            self.grads[i] = random_weight_matrix(dim[0],dim[1])
                   
        x = []
        #print self.wv[0]
        #print self.grads

        for i in window:
            x.extend(self.wv[i,])
            
        x = matrix(x)

        ##
        # Forward propagation
        
        #print self.grads.W.shape
        print 'x_shape',x.shape
        #print self.grads.b1.shape

        z1 = dot(self.grads.W,x.transpose())
        #print z1.shape
        z1 = (z1.transpose() + matrix(self.grads.b1)).transpose()
        h = 2*self.sigmoid(2*z1) - 1
        h = matrix(h)
        print"u_Shape", self.grads.W.shape
        #print h.shape

        z2 = dot(self.grads.U,h)
        z2 = (z2.transpose() + matrix(self.grads.b2)).transpose()
        y = softmax(z2)

        ##
        # Backpropagation
        #print label,y
        new_label = [0,0,0,0,0]
        new_label[label] = 1

        delta2 = y.transpose() - new_label
        temp = dot(h,delta2)
        #print temp.shape,self.grads.U.shape
        #print 'del2',delta2.shape
        self.grads.U +=  temp.transpose()
        self.grads.b2 = matrix(self.grads.b2)+delta2

        temp = dot(delta2,self.grads.U)
        #print 'temp,h',temp.shape,h.shape
        temp = multiply(self.tan_grad(h),temp.transpose())
        print 'temp,b1',temp.shape,matrix(self.grads.b1).shape
        
        self.grads.b1 = matrix(self.grads.b1)+temp.transpose()
        temp2 = dot(temp.transpose(),self.grads.W)
        temp = dot(temp,x)
        self.grads.W += temp

        print self.sgrads.L

        

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####


        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####


        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####


        #### END YOUR CODE ####
        return J
