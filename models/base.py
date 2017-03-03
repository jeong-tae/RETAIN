# parent class of all variation of RETAIN

class base(object):
    def __init__(self, config):
    """ Abstaction of all medical models in this project """
        self.batch_size = config.batch_size
        self.n_steps = config.n_steps
        self.input_dim = config.input_dim
        self.target_dim = config.target_dim
        self.m_dim = config.m_dim
        self.p_dim = config.p_dim

    @property
    def new_cell(self, dim, rnn_type = 'lstm', is_tuple = True):
        if rnn_type.lower() == 'lstm':
            return tf.nn.rnn_cell.LSTMCell(dim, state_is_tuple = True)
        elif rnn_type.lower() == 'gru':
            return tf.nn.rnn_Cell.GRUCell(dim)
        else:
            raise ValueError(" [E] Doesn't supported rnn_type: %s"%rnn_type)

