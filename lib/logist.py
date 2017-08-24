"""
    Implementation of baseline LogistRegression
    Xiaopeng LI
    Mar. 20, 2017@HKUST
"""
import numpy as np
import utils as utils
import sys
import math
import scipy
import scipy.io
import logging
import optim
from sklearn.metrics import roc_auc_score

class Params:
    """Parameters for RDL
    """
    def __init__(self):
        self.lambda_w = 1e-4
        self.lambda_v = 0.1
        self.lambda_n = 1
        self.lambda_e = 0.001

        # for updating W and b
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 128

        self.n_epochs = 10

def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp( -x ) )

class LogistRegression:
    """LogistRegression"""
    def __init__(self, num_items, num_factors, params, print_step=50, verbose=True):
        self.m_num_items = num_items
        self.m_num_factors = num_factors

        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.eta = np.random.randn(self.m_num_factors)

        self.params = params
        self.print_step = print_step
        self.verbose = verbose

        # for v parameters
        # construct adam optimizer for each v_i
        # Make sure the update rule exists, then replace the string
        # name with the actual function
        self.update = "adam"
        if not hasattr(optim, self.update):
          raise ValueError('Invalid update_rule "%s"' % self.update)
        self.update = getattr(optim, self.update)

        self.v_configs = [None] * self.m_num_items
        optim_config={'learning_rate': self.params.lr}
        for i in range(self.m_num_items):
            d = {k: v for k, v in optim_config.iteritems()}
            self.v_configs[i] = d

        # adam optimizer for eta parameter
        d = {k: v for k, v in optim_config.iteritems()}
        self.eta_config = d

    def latent_estimate(self, links):
        # links is a list. Each element is an array of ids linking to the item
        # gradient descent for each v
        likelihood = 0
        for i in range(self.m_num_items):
            link_ids = links[i]
            if len(link_ids) == 0:
                self.m_V[i][:] = self.m_theta[i]
            link_v = self.m_V[link_ids]
            v = self.m_V[i]
            theta = self.m_theta[i]
            gv = -np.dot((1-sigmoid(np.sum((v*link_v)*self.eta, axis=1))), self.eta*link_v) \
                + self.params.lambda_v * (v - theta)
            next_v, config = self.update(v, gv, self.v_configs[i])
            self.m_V[i][:] = next_v
            self.v_configs[i] = config
            likelihood += -0.5 * self.params.lambda_v *np.sum(np.square(self.m_V[i] - self.m_theta[i]))

        # gradient descent for eta
        g_eta = np.zeros_like(self.eta)
        likelihood_link = 0
        for i in range(self.m_num_items):
            link_ids = links[i]
            if len(link_ids) == 0:
                continue
            link_v = self.m_V[link_ids]
            v = self.m_V[i]
            theta = self.m_theta[i]
            g_eta += -np.dot((1-sigmoid(np.sum((v*link_v)*self.eta, axis=1))), v*link_v)
            likelihood_link += np.sum(np.log(sigmoid(np.sum( (v*link_v)*self.eta, axis=1))))
        likelihood_link = 1.0 * likelihood_link/2
        g_eta[:] = g_eta/2 + self.params.lambda_e * self.eta
        next_eta, config = self.update(self.eta, g_eta, self.eta_config)
        self.eta[:] = next_eta
        self.eta_config = config

        # estimate the likelihood
        likelihood += likelihood_link - 0.5 * self.params.lambda_e * np.sum(np.square(self.eta))
        return likelihood

    def predict(self, test_links):
        sum_rank = 0
        num_links = 0
        auc = []
        for i in range(self.m_num_items):
            link_ids = test_links[i]
            if len(link_ids) == 0:
                continue
            v = self.m_V[i]
            # probs = sigmoid(np.sum(self.eta * (v * self.m_V), axis=1))
            probs = np.zeros(self.m_num_items)
            for adi in xrange(self.m_num_items):
                probs[adi] = np.dot(self.eta, self.m_V[i] * self.m_V[adi])
            y_true = np.zeros(self.m_num_items)
            y_true[link_ids] = 1
            # compute link rank
            ordered = probs.argsort()[::-1].tolist() # descending order
            ranks = [ordered.index(x) for x in link_ids]
            sum_rank += sum(ranks)
            num_links += len(ranks)
            # compute auc
            y_score = np.delete(probs, i)
            y_true = np.delete(y_true, i)
            auc.append( roc_auc_score(y_true, y_score) )
        ave_rank = 1.0 * sum_rank/num_links
        ave_auc = np.mean(np.array(auc))
        return (ave_rank, ave_auc)

    def run(self, data_x, links, test_links):
        self.m_theta[:] = data_x
        self.m_V[:] = self.m_theta
        n = data_x.shape[0]
        for epoch in range(self.params.n_epochs):
            likelihood = self.latent_estimate(links)
            ave_rank, ave_auc = self.predict(test_links)
            loss = -likelihood
            logging.info("[#epoch=%06d], loss=%.5f, ave_rank=%.4f, ave_auc=%.4f" % (
                epoch, loss, ave_rank, ave_auc))
        scipy.io.savemat("temp/model-"+str(epoch)+".mat",{"m_V": self.m_V, "m_theta": self.m_theta})

