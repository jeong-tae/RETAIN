# make RETAIN code

from base import base
import tensorflow as tf
import numpy as np

class RETAIN(base):
    def __init__(self, config):
        super(RETAIN, self).__init__(config)

        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.n_steps])

    def length(data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices = 2)
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length

    def build_model(self):
        
        with tf.variable_scope('emb'):
            W_emb = tf.get_variable('weight', shape = [self.input_dim, self.m_dim])

        v_i = []
        for i in range(self.n_steps):
            v = tf.matmul(self.X[:, i, :], W_emb)
            v = tf.reshape(v, [-1, 1, self.m_dim])
            v_i.append(v)
        v_i = tf.concat(v_i, axis = 1)

        rnn_alpha = self.new_cell(self.p_dim, 'lstm', is_tuple = True)
        # set RNN layer to improve, such as stacking the layer, Dropout, ...

        reverse_v = tf.reverse(v_i, dims = [False, True, False])

        """
        reverse_v = []
        for i in range(self.n_steps, 0, -1):
            reverse_v.append(v_i[:, i-1, :])
        reverse_v = tf.concat(1, reverse_v)
        """

        reverse_g_i, alpha_states = tf.nn.dynamic_rnn(rnn_alpha, reverse_v, sequence_length=self.length(self.X), scope = 'rnn_alpha')

        g_i = tf.reverse(reverse_g_i, dims = [False, True, False])
        
        """
        g_i = []
        for i in range(self.n_steps, 0, -1):
            g_i.append(reverse_g_i[:, i-1, :])
        g_i = tf.concat(1, g_i)
        """

        # attention parameter to learn
        with tf.variable_scope('alpha'):
            W_alpha = tf.get_variable('weight', shape = [self.p_dim, 1])
            b_alpha = tf.get_variable('bias', shape = [1])
        
        # attention
        e_j = []
        for j in range(self.n_steps):
            e = tf.matmul(g_i[:, j, :], W_alpha) + b_alpha
            e_j.append(e)
        alpha_i = tf.nn.softmax(tf.concat(1, e_j), dim = 1)

        rnn_beta = self.new_cell(self.q_dim, 'lstm', is_tuple = True)
        # need more detail of RNN configuration        

        reverse_h_i, beta_states = tf.nn.dynamic_rnn(rnn_beta, reverse_v, sequence_length=self.length(self.X), scope = 'rnn_beta')

        h_i = tf.reverse(reverse_h_i, dims = [False, True, False])

        """
        h_i = []
        for i in range(self.n_steps, 0, -1):
            h_i.append(reverse_h_i[:, i-1, :])
        h_i = tf.concat(1, h_i)
        """

        with tf.variable_scope('beta'):
            W_beta = tf.get_variable('weight', shape = [self.q_dim, self.m_dim])
            b_beta = tf.get_variable('bias', shape = [self.m_dim])

        beta_j = []
        for j in range(self.n_steps):
            beta = tf.tanh(tf.matmul(h_i[:, j, :], W_beta) + b_beta)
            beta = tf.reshape(beta, [-1, 1, self.m_dim])
            beta_j.append(beta)
        beta_j = tf.concat(beta_j, axis = 1)

        c_i = []
        for i in range(self.n_steps):
            alpha_beta_v = []
            for j in range(i+1):
                attention = alpha_i[:, j] * tf.multiply(beta_j[:, j, :], v_i[:, j, :])
                attention = tf.reshape(attention, [-1, 1, self.m_dim])
                alpha_beta_v.append(attetion)
            alpha_beta_v = tf.concat(alpha_beta_v, axis = 1)
            c = tf.reduce_sum(alpha_beta_v, reduction_indices = 1)
            c = tf.reshape(c, [-1, 1, self.m_dim])
            c_i.append(c)
        c_i = tf.concat(c_i, axis = 1)

        with tf.variable_scope('logit'):
            W_logit = tf.get_variable('weight', shape = [self.m_dim, self.target_dim])
            b_logit = tf.get_variable('bias', shape = [self.target_dim])

        c_i_flat = tf.reshape(c_i, [-1, self.m_dim])
        self.logits_flat = tf.batch_matmul(c_i_flat, W_logit) + b_logit
        self.y_hat_i = tf.reshape(tf.nn.softmax(self.logits), [-1, self.n_steps, self.target_dim])

        contributions = []
        for i in range(self.n_steps):
            coef = alpha_i[:, j] * tf.matmul(tf.transpose(W_logit), tf.multiply(beta_j[:, j, :], tf.transpose(W_emb)))
            coef = tf.reshape(coef, [-1, 1, self.target_dim, self.input_dim])
            contributions.append(coef)
        contributions = tf.concat(contributions, axis = 1)
        # if you want to see i'th visit, c'th disease of contribution
        # then use contribution[:, i, c, :].
        
    def optimization(self):
        if self.Y_is_onehot == True:
            # need to define this
        else:
            Y_flat = tf.reshape(y, [-1])
            if learning_type = 'logistic':
                self.losses = tf.nn.sparse_sigmoid_cross_entropy_with_logits(self.logits_flat, Y_flat)
            else:
                self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat, Y_flat)
            mask = tf.sign(tf.reduce_max(tf.abs(self.X), reduction_indices=2)
            mask_flat = tf.reshape(mask, [-1])
            masked_losses = mask_float * self.losses
            masekd_losses = tf.reshape(masked_losses, tf.shape(self.Y))
            # should X and Y paddings are aligned with, if not the code make bug

            mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / self.length(self.X)
            self.mean_loss = tf.reduce_mean(mean_loss_by_example)

    def run_iteration(self):
        # get batch_size of data


    def run(self):
        # fit the model

