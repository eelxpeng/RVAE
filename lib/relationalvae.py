import numpy as np
import utils as utils
import tensorflow as tf
import sys
import math
import scipy
import scipy.io
import logging
import optim
from sklearn.metrics import roc_auc_score

class Params:
    """Parameters for RVAE
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

class RelationalVAE:
    def __init__(self, num_items, num_factors, params, input_dim, 
        dims, activations, n_z=50, loss_type='cross-entropy',
        random_seed=0, print_step=50, verbose=True):
        self.m_num_items = num_items
        self.m_num_factors = num_factors

        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)

        self.eta = 0.1 * np.random.randn(self.m_num_factors)

        self.input_dim = input_dim
        self.dims = dims
        self.activations = activations
        self.params = params
        self.print_step = print_step
        self.verbose = verbose
        self.n_z = n_z
        self.weights = []
        self.reg_loss = 0

        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        # self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])

        # now assuming accumulating gradient one by one
        self.linked_v = tf.placeholder(tf.float32, [None, self.m_num_factors])
        self.eta_vae = tf.placeholder(tf.float32,[self.m_num_factors])

        x_recon = self.inference_generation(self.x)

        # loss
        # reconstruction loss
        if loss_type == 'rmse':
            self.gen_loss = tf.reduce_mean(tf.square(tf.sub(self.x, x_recon)))
        elif loss_type == 'cross-entropy':
            x_recon = tf.nn.sigmoid(x_recon, name='x_recon')
            # self.gen_loss = -tf.reduce_mean(self.x * tf.log(tf.maximum(x_recon, 1e-10)) 
            #     + (1-self.x)*tf.log(tf.maximum(1-x_recon, 1e-10)))
            self.gen_loss = -tf.reduce_mean(tf.reduce_sum(self.x * tf.log(tf.maximum(x_recon, 1e-10)) 
                + (1-self.x) * tf.log(tf.maximum(1 - x_recon, 1e-10)),1))

        self.latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq)
            - self.z_log_sigma_sq - 1, 1))
        # self.v_loss = 1.0*params.lambda_v/params.lambda_n * tf.reduce_mean( tf.reduce_sum(tf.square(self.v - self.z), 1))
        self.link_loss = -tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(self.eta_vae*self.z*self.linked_v, 1))))

        self.loss = self.gen_loss + self.latent_loss + self.link_loss
        self.optimizer = tf.train.AdamOptimizer(self.params.lr).minimize(self.loss)

        self.tvars = tf.trainable_variables()
        self.newGrads = tf.gradients(self.loss,self.tvars)
        self.regGrads = tf.gradients(self.reg_loss, self.tvars)
        adam = tf.train.AdamOptimizer(learning_rate=self.params.lr)
        self.updateGrads = adam.apply_gradients(zip(self.batchGrad,self.tvars))

        # Initializing the tensor flow variables
        self.saver = tf.train.Saver(self.weights)
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)

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

    def inference_generation(self, x):
        with tf.variable_scope("inference"):
            rec = {'W1': tf.get_variable("W1", [self.input_dim, self.dims[0]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b1': tf.get_variable("b1", [self.dims[0]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W2': tf.get_variable("W2", [self.dims[0], self.dims[1]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2", [self.dims[1]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_mean': tf.get_variable("W_z_mean", [self.dims[1], self.n_z], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_mean': tf.get_variable("b_z_mean", [self.n_z], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.dims[1], self.n_z], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}
        self.weights += [rec['W1'], rec['b1'], rec['W2'], rec['b2'], rec['W_z_mean'],
                        rec['b_z_mean'], rec['W_z_log_sigma'], rec['b_z_log_sigma']]
        self.reg_loss += tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2']) + tf.nn.l2_loss(rec["b1"]) + \
                        tf.nn.l2_loss(rec["b2"]) + tf.nn.l2_loss(rec["W_z_mean"]) + tf.nn.l2_loss(rec["b_z_mean"]) +\
                        tf.nn.l2_loss(rec["W_z_log_sigma"]) + tf.nn.l2_loss(rec["b_z_log_sigma"])

        W1Grad = tf.placeholder(tf.float32, name="W1_grad")
        b1Grad = tf.placeholder(tf.float32, name="b1_grad")
        W2Grad = tf.placeholder(tf.float32, name="W2_grad")
        b2Grad = tf.placeholder(tf.float32, name="b2_grad")
        W_z_mean_Grad = tf.placeholder(tf.float32, name="W_z_mean_Grad")
        b_z_mean_Grad = tf.placeholder(tf.float32, name="b_z_mean_Grad")
        W_z_log_sigma_Grad = tf.placeholder(tf.float32, name="W_z_log_sigma_Grad")
        b_z_log_sigma_Grad = tf.placeholder(tf.float32, name="b_z_log_sigma_Grad")
        self.batchGrad = []
        self.batchGrad += [W1Grad, b1Grad, W2Grad, b2Grad, W_z_mean_Grad,
                            b_z_mean_Grad, W_z_log_sigma_Grad, b_z_log_sigma_Grad]

        h1 = self.activate(
            tf.matmul(x, rec['W1']) + rec['b1'], self.activations[0])
        h2 = self.activate(
            tf.matmul(h1, rec['W2']) + rec['b2'], self.activations[1])
        self.z_mean = tf.matmul(h2, rec['W_z_mean']) + rec['b_z_mean']
        self.z_log_sigma_sq = tf.matmul(h2, rec['W_z_log_sigma']) + rec['b_z_log_sigma']

        eps = tf.random_normal((1, self.n_z), 0, 1, 
            seed=0, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps

        with tf.variable_scope("generation"):
            gen = {'W2': tf.get_variable("W2", [self.n_z, self.dims[1]], 
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                'b2': tf.get_variable("b2", [self.dims[1]], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                'W1': tf.transpose(rec['W2']),
                'b1': rec['b1'],
                'W_x': tf.transpose(rec['W1']),
                'b_x': tf.get_variable("b_x", [self.input_dim], 
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        self.weights += [gen['W2'], gen['b2'], gen['b_x']]
        self.reg_loss += tf.nn.l2_loss(gen['W2']) + tf.nn.l2_loss(gen['W_x']) + \
                        tf.nn.l2_loss(gen["b2"]) + tf.nn.l2_loss(gen["b_x"])
        h2 = self.activate(
            tf.matmul(self.z, gen['W2']) + gen['b2'], self.activations[1])
        h1 = self.activate(
            tf.matmul(h2, gen['W1']) + gen['b1'], self.activations[0])
        x_recon = tf.matmul(h1, gen['W_x']) + gen['b_x']

        W2Grad_Gen = tf.placeholder(tf.float32, name="W2_grad_gen")
        b2Grad_Gen = tf.placeholder(tf.float32, name="b2_grad_gen")
        b_xGrad_Gen = tf.placeholder(tf.float32, name="b_x_grad_gen")

        self.batchGrad += [W2Grad_Gen, b2Grad_Gen, b_xGrad_Gen]

        return x_recon

    def rvae_estimate(self, data_x, links, num_iter):
        gradBuffer = self.sess.run(self.tvars)
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        for i in range(num_iter):
            b_x, ids = utils.get_batch(data_x, self.params.batch_size)
            num = 0
            gen_loss = 0
            for j in range(self.params.batch_size):
                x = b_x[j].reshape((1,-1))
                id = ids[j]
                link_ids = links[id]
                if len(link_ids) == 0:
                    continue
                link_v = self.m_V[link_ids]
                tGrad, gen_loss_ins = self.sess.run((self.newGrads, self.gen_loss),
                    feed_dict={self.x: x, self.linked_v: link_v, self.eta_vae: self.eta})
                gen_loss += gen_loss_ins
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                num += 1
            gen_loss = gen_loss/num
            tGrad = self.sess.run(self.regGrads)
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += gradBuffer[ix]/num + grad * self.params.lambda_w

            feed_dict = {}
            for j in range(len(self.batchGrad)):
                feed_dict[self.batchGrad[j]] = gradBuffer[j]
            self.sess.run(self.updateGrads,feed_dict=feed_dict)
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

            # Display logs per epoch step
            if i % self.print_step == 0 and self.verbose:
                print "Iter:", '%04d' % (i+1), \
                      "loss=", "{:.5f}".format(l), \
                      "genloss=", "{:.5f}".format(gen_loss), \
                      "vloss=", "{:.5f}".format(v_loss)
        return gen_loss

    def transform(self, data_x):
        data_en = self.sess.run(self.z_mean, feed_dict={self.x: data_x})
        return data_en

    def latent_estimate(self, links):
        # links is a list. Each element is an array of ids linking to the item
        # gradient descent for each v
        likelihood = 0

        # gradient descent for eta
        g_eta = np.zeros_like(self.eta)
        likelihood_link = 0
        for i in range(self.m_num_items):
            link_ids = links[i]
            if len(link_ids) == 0:
                continue
            link_v = self.m_V[link_ids]
            v = self.m_V[i]
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

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def run(self, data_x, links, test_links):
        self.m_V[:] = self.transform(data_x)
        n = data_x.shape[0]
        perm_idx = np.random.permutation(n)
        for epoch in range(self.params.n_epochs):
            num_iter = int(n / self.params.batch_size)
            gen_loss = self.rvae_estimate(data_x, links, num_iter)
            self.m_V[:] = self.transform(data_x)
            likelihood = self.latent_estimate(links)
            ave_rank, ave_auc = self.predict(test_links)
            loss = -likelihood + 0.5 * gen_loss * n * self.params.lambda_n
            logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f, ave_rank=%.4f, ave_auc=%.4f" % (
                epoch, loss, -likelihood, gen_loss, ave_rank, ave_auc))

    def save_model(self, weight_path, pmf_path=None):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)
        if pmf_path is not None:
            scipy.io.savemat(pmf_path,{"m_V": self.m_V, "m_theta": self.m_theta})
            logging.info("Weights saved at " + pmf_path)

    def load_model(self, weight_path, pmf_path=None):
        logging.info("Loading weights from " + weight_path)
        self.saver.restore(self.sess, weight_path)
        if pmf_path is not None:
            logging.info("Loading pmf data from " + pmf_path)
            data = scipy.io.loadmat(pmf_path)
            self.m_V[:] = data["m_V"]
            self.m_theta[:] = data["m_theta"]

