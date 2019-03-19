from keras import optimizers
from keras.legacy import interfaces
from keras import backend as K

class RAME(optimizers.Optimizer):
    """Rapidly adapting moment estimation (RAME)
        The code for RAME is inherited from the original implementation of SGD in Keras 

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float > 0. 
        decay: float >= 0. Learning rate decay over each update.
    """

    def __init__(self, lr=0.01, momentum=0.9, quantum = 0.25, decay=0., **kwargs):
        super(RAME, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.quantum = K.variable(quantum, name='quantum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m + lr * g  # velocity
            self.updates.append(K.update(m, v))

            new_p = p - K.sign(v)*K.pow(K.abs(v), 1-self.quantum) 

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'quantum': float(K.get_value(self.quantum)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(RAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

