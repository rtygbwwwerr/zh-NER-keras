'''
author: liaoyixuan
time: 2019/8/31 11:25
desc:
'''


from keras.layers import RNN, Dense, Reshape
from keras.layers.recurrent import LSTMCell
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.activations import softmax
from keras.layers import Layer
import tensorflow as tf


ALLOWERED_KWARGS = {'input_shape',
          'batch_input_shape',
          'batch_size',
          'dtype',
          'name',
          'trainable',
          'weights',
          'input_dtype',  # legacy
          }

class BasicLSTM(RNN):
    """Basic Long Short-Term Memory layer - vivo AILab shenzheng 2019.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = self.create_cell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation,
                        **kwargs)
        
        super(BasicLSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **self.clear_args(kwargs))
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def clear_args(self, args):
        filter_args = {}
        for k, v in args.items():
            if k in ALLOWERED_KWARGS:
                filter_args[k] = v
        return filter_args
        
    def create_cell(self, units,
                        activation,
                        recurrent_activation,
                        use_bias,
                        kernel_initializer,
                        recurrent_initializer,
                        unit_forget_bias,
                        bias_initializer,
                        kernel_regularizer,
                        recurrent_regularizer,
                        bias_regularizer,
                        kernel_constraint,
                        recurrent_constraint,
                        bias_constraint,
                        dropout,
                        recurrent_dropout,
                        implementation,
                        **kwargs):
        raise("BasicLSTM is not a implemented class, you need to override the create_cell method")
        
        return None
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        '''
        inputs: [batch_size, sentence_length, embedding_dim]
        '''

        return super(BasicLSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units
    
    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(BasicLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


class MultiInputLSTM(BasicLSTM):
    """Multi-input Long Short-Term Memory layer - vivo AILab shenzheng 2019.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        input_lengths:list of length for each input vector--[len1, len2,...]
        input_len:length of final input
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def create_cell(self, units,
                        activation,
                        recurrent_activation,
                        use_bias,
                        kernel_initializer,
                        recurrent_initializer,
                        unit_forget_bias,
                        bias_initializer,
                        kernel_regularizer,
                        recurrent_regularizer,
                        bias_regularizer,
                        kernel_constraint,
                        recurrent_constraint,
                        bias_constraint,
                        dropout,
                        recurrent_dropout,
                        implementation,
                        **kwargs):
        
        if "input_len" not in kwargs:
            kwargs["input_len"] = None
        if "input_lengths" not in kwargs:
            kwargs['input_lengths'] = None
            
        return MultiLSTMCell(units,
                        input_lengths=kwargs['input_lengths'],
                        input_len=kwargs['input_len'],
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)


    
    @property
    def input_lengths(self):
        return self.cell.input_lengths
    
    @property
    def input_len(self):
        return self.cell.input_len
    


    def get_config(self):
        config = {
                  'input_lengths': self.cell.input_lengths,
                  'input_len': self.cell.input_len,
                  }
        base_config = super(MultiInputLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class MultiInputCellWapper(Layer):
#     def __init__(self, Layer):
        



class MultiLSTMCell(LSTMCell):
    """Cell class for the LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        input_lengths:list of length for each input vector--[len1, len2,...]
        input_len:length of final input
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 input_lengths,
                 input_len=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        
        super(MultiLSTMCell, self).__init__(
                 units=units,
                 activation=activation,
                 recurrent_activation=recurrent_activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 recurrent_initializer=recurrent_initializer,
                 bias_initializer=bias_initializer,
                 unit_forget_bias=unit_forget_bias,
                 kernel_regularizer=kernel_regularizer,
                 recurrent_regularizer=recurrent_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 recurrent_constraint=recurrent_constraint,
                 bias_constraint=bias_constraint,
                 dropout=dropout,
                 recurrent_dropout=recurrent_dropout,
                 implementation=implementation,
                 **kwargs)
        
        self.input_lengths = input_lengths
        self.input_len = input_len

        if input_lengths is not None:
            self.input_len = max(input_lengths)
        
    def build(self, input_shape):
        
        if self.input_lengths is not None:
            self.kernel_inp_att = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel_inp_att',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        if self.input_len is not None:
            super(MultiLSTMCell, self).build(input_shape=((input_shape[:-1] + (self.input_len,))))
        else:
            super(MultiLSTMCell, self).build(input_shape=input_shape)
        
    def reweight(self, inputs, c_tm1):
        """
        imposing input attention on multi-column inputs
        weights=attention([input1, input2,...], c(t-1))
        input = inputs * weights
        """
        if self.input_lengths is None:
            return inputs
        
        xs = tf.split(inputs, self.input_lengths, -1)
        z = []
        start = 0
        for i in range(len(self.input_lengths)):
            end = start + self.input_lengths[i]
            w = self.kernel_inp_att[start:end, :]
            #x * w * c(t-1)
            mul = K.dot(xs[i], w)
            mul = K.batch_dot(mul, c_tm1, axes=[1,1])

            z.append(mul)
            start = end
            
        z = tf.concat(z, axis=1)
        weights = softmax(z)
        xs = tf.stack(xs, axis=1)
        x = K.batch_dot(weights, xs, axes=[1, 1])
#         tf.summary.histogram("summary_name", weights)
        return x
        

    def call(self, inputs, states, training=None):
        '''
        inputs: [batch_size, embedding_dim]
        states: [h(t-1), c(t-1)]
        '''
        c_tm1 = states[1]
        inputs = self.reweight(inputs, c_tm1)
        return super(MultiLSTMCell, self).call(inputs, states, training)
        

    def get_config(self):
        config = {
                  'input_lengths': self.input_lengths,
                  'input_len': self.input_len
                  }
        base_config = super(MultiLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
class AttentionLSTM(BasicLSTM):
    """attention Long Short-Term Memory layer - vivo AILab shenzheng 2019.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        input_lengths:list of length for each input vector--[len1, len2,...]
        input_len:length of final input
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def create_cell(self, units,
                        activation,
                        recurrent_activation,
                        use_bias,
                        kernel_initializer,
                        recurrent_initializer,
                        unit_forget_bias,
                        bias_initializer,
                        kernel_regularizer,
                        recurrent_regularizer,
                        bias_regularizer,
                        kernel_constraint,
                        recurrent_constraint,
                        bias_constraint,
                        dropout,
                        recurrent_dropout,
                        implementation,
                        **kwargs):
        
        if "context" not in kwargs:
            kwargs["context"] = None
        if "context_length" not in kwargs:
            kwargs['context_length'] = None
        if "att_hidden_size" not in kwargs:
            kwargs['att_hidden_size'] = None
            
        return AttLSTMCell(units,
                        context=kwargs['context'],
                        context_length=kwargs['context_length'],
                        att_hidden_size=kwargs['att_hidden_size'],
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)


    
    @property
    def context(self):
        return self.cell.context
    
    @property
    def context_length(self):
        return self.cell.context_length
    
    @property
    def att_hiden_size(self):
        return self.cell.att_hiden_size

    def get_config(self):
        config = {
                  'context': self.cell.context,
                  'context_length': self.cell.context_length,
                  'att_hiden_size': self.cell.att_hiden_size,
                  }
        base_config = super(AttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
class AttLSTMCell(LSTMCell):
    """attention Cell class for the LSTM layer.
    implement type:Bahdanau
    # Arguments
        units: Positive integer, dimensionality of the output space.
        context:[batch_size, sentence_length, embedding_dim]
        input_len:length of final input
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 context,
                 context_length,
                 att_hidden_size,
                 att_type="bahdanau",
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        
        super(AttLSTMCell, self).__init__(
                 units=units,
                 activation=activation,
                 recurrent_activation=recurrent_activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 recurrent_initializer=recurrent_initializer,
                 bias_initializer=bias_initializer,
                 unit_forget_bias=unit_forget_bias,
                 kernel_regularizer=kernel_regularizer,
                 recurrent_regularizer=recurrent_regularizer,
                 bias_regularizer=bias_regularizer,
                 kernel_constraint=kernel_constraint,
                 recurrent_constraint=recurrent_constraint,
                 bias_constraint=bias_constraint,
                 dropout=dropout,
                 recurrent_dropout=recurrent_dropout,
                 implementation=implementation,
                 **kwargs)
        
        self.context = context
        self.context_length = context_length
        self.att_hidden_size = att_hidden_size
        self.att_type = att_type

    def reset_states(self):
        self.att_hidden_layer.reset_states()
        self.att_output_layer.reset_states()
        super(AttLSTMCell, self).reset_states()
        
#     def reuse(self, layer, *args, **kwargs):
#         if not layer.built:
#             if len(args) > 0:
#                 inputs = args[0]
#             else:
#                 inputs = kwargs['inputs']
#             if isinstance(inputs, list):
#                 input_shape = [K.int_shape(x) for x in inputs]
#             else:
#                 input_shape = K.int_shape(inputs)
#             layer.build(input_shape)
#         outputs = layer.call(*args, **kwargs)
#         for w in layer.trainable_weights:
#             if w not in self._trainable_weights:
#                 self._trainable_weights.append(w)
#         for w in layer.non_trainable_weights:
#             if w not in self._non_trainable_weights:
#                 self._non_trainable_weights.append(w)
#         for u in layer.updates:
#             if not hasattr(self, '_updates'):
#                 self._updates = []
#             if u not in self._updates:
#                 self._updates.append(u)
#         return outputs
    
    def build(self, input_shape):
        
        att_input_size = input_shape[-1] + self.units
        
#         self.kernel_att_hidden = self.add_weight(shape=(att_input_size, self.att_hidden_size),
#                                   name='kernel_att',
#                                   initializer=self.kernel_initializer,
#                                   regularizer=self.kernel_regularizer,
#                                   constraint=self.kernel_constraint)
#         
#         self.kernel_att_out = self.add_weight(shape=(self.att_hidden_size, self.context_length),
#                                   name='kernel_att',
#                                   initializer=self.kernel_initializer,
#                                   regularizer=self.kernel_regularizer,
#                                   constraint=self.kernel_constraint)
#         self.att_input = Input()
        self.att_hidden_layer = Dense(self.att_hidden_size, activation='relu', name="att_hidden_layer")
        self.att_output_layer = Dense(self.context_length, activation='softmax', name="att_output_layer")
        self.att_reshape_layer = Reshape((self.context_length * self.att_hidden_size,))
        
        with K.name_scope(self.att_hidden_layer.name):
            self.att_hidden_layer.build(input_shape)
            
        with K.name_scope(self.att_output_layer.name):
            self.att_output_layer.build(input_shape)
            
        with K.name_scope(self.att_reshape_layer.name):
            self.att_reshape_layer.build(input_shape)
            
        super(AttLSTMCell, self).build(input_shape=input_shape)
        self.built = True
        
    def attention(self, inputs, s_tm1):
        """
        :param inputs:[batch_size, input_dim]
        :param s_tm1:[batch_size, units_num]
        imposing attention on encoder embedding
        weights=attention([input1, input2,...], c(t-1))
        input = inputs * weights
        """
   
        
        if self.att_type == "bahdanau":
            
            x = self.attention_bahdanau(self.context, s_tm1)
        elif self.att_type == "cosine":
            x = self.attention_cosine(self.context, s_tm1)
        else:
            raise("unsupported attention type:{}".format(self.att_type))
#         tf.summary.histogram("summary_name", weights)
        return x
    
    def attention_bahdanau(self, context, s_tm1):
        s_tm1_seq = K.repeat(s_tm1, self.context_length)
        att_x = K.concatenate([context, s_tm1_seq])
        
        att_hidden = self.att_hidden_layer(att_x)
        att_hidden = self.att_reshape_layer(att_hidden)
        weights = self.att_output_layer(att_hidden)
        x = K.batch_dot(weights, context, axes=[1, 1])
        return x
    
    def attention_cosine(self, context, s_tm1):

        return x

    def call(self, inputs, states, training=None):
        '''
        inputs: [batch_size, embedding_dim]
        states: [h(t-1), c(t-1)]
        '''
        s_tm1 = states[0]
        inputs = self.attention(inputs, s_tm1)
        return super(AttLSTMCell, self).call(inputs, states, training)
    
    def get_weights(self):
        return self.att_hidden_layer.get_weights() + self.att_output_layer.get_weights()
    
    @property
    def trainable_weights(self):
        weights = []
        
        if hasattr(super(AttLSTMCell, self), 'trainable_weights'):
            weights += super(AttLSTMCell, self).trainable_weights
        
        if hasattr(self.att_hidden_layer, 'trainable_weights'):
            weights += self.att_hidden_layer.trainable_weights
            
        if hasattr(self.att_output_layer, 'trainable_weights'):
            weights += self.att_output_layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        if hasattr(super(AttLSTMCell, self), 'non_trainable_weights'):
            weights += super(AttLSTMCell, self).non_trainable_weights
        
        if hasattr(self.att_hidden_layer, 'non_trainable_weights'):
            weights += self.att_hidden_layer.non_trainable_weights
            
        if hasattr(self.att_output_layer, 'non_trainable_weights'):
            weights += self.att_output_layer.non_trainable_weights
        return weights

    @property
    def updates(self):
        updates = []
        
        if hasattr(super(AttLSTMCell, self), 'updates'):
            weights += super(AttLSTMCell, self).updates
            
        if hasattr(self.att_hidden_layer, 'updates'):
            updates += self.att_hidden_layer.updates
            
        if hasattr(self.att_output_layer, 'updates'):
            updates += self.att_output_layer.updates
        return updates
        
    
    def get_config(self):
        config = {
                  'input_lengths': self.input_lengths,
                  'input_len': self.input_len
                  }
        base_config = super(MultiLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
