import tensorflow as tf
from .UMNN import ParallelNeuralIntegral

def _flatten(sequence):
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, axis=0) if len(flat) > 0 else tf.constant([])


class IntegrandNN(tf.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.layers = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.layers.append(tf.keras.layers.Dense(h1))
            self.layers.append(tf.keras.layers.ReLU())
        self.layers.pop() # 移除最后一个ReLU
        self.layers.append(tf.keras.layers.ELU())

    def __call__(self, x, h):
        net_input = tf.concat([x, h], axis=1)
        net_output = net_input
        for layer in self.layers:
            net_output = layer(net_output)
        return net_output + 1.0
    
class MonotonicNN(tf.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.layers = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.layers.append(tf.keras.layers.Dense(h1))
            self.layers.append(tf.keras.layers.ReLU())
        self.layers.pop() # 移除最后一个ReLU
        # 输出缩放和偏移因子
        self.nb_steps = nb_steps

    def __call__(self, x, h):
        x0 = tf.zeros_like(x)
        net_output = h
        for layer in self.layers:
            net_output = layer(net_output)
        # offset = tf.expand_dims(net_output[:, 0], axis=1)
        # scaling = tf.exp(tf.expand_dims(net_output[:, 1], axis=1))
        
        integral = ParallelNeuralIntegral()
        x_tot = integral.forward(x0, x, self.integrand, _flatten(self.integrand.trainable_variables), h, self.nb_steps)
        
        return x_tot 
