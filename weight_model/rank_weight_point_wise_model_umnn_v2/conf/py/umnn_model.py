import tensorflow as tf
import numpy as np
import math

def compute_cc_weights(nb_steps):
    # 生成Clenshaw-Curtis积分权重和节点 
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] = 0.5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = tf.constant(lam.T @ W, dtype=tf.float32)
    steps = tf.constant(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps), dtype=tf.float32)
    return cc_weights, steps

class IntegrandNN(object):
    def __init__(self, in_d, hidden_layers, inv_f=False):
        # 构建被积函数神经网络
        self.layers = []
        self.inv_f = inv_f
        self.in_d = in_d
        self.hidden_layers = hidden_layers
        # hs = [in_d] + hidden_layers + [1]
        # with tf.variable_scope("IntegrandNN"):
        #     for i, h1 in enumerate(hs):
        #         self.layers.append(tf.layers.Dense(h1, activation=tf.nn.relu if i < len(hs)-2 else None))
        #     self.output_activation = tf.nn.elu  # 最后使用ELU激活

    def unmm_network(self, name, inputs, units, act=tf.nn.relu, last_act=tf.nn.elu, stop_gradient=False):
        if stop_gradient:
            output = tf.stop_gradient(inputs)
        else:
            output = inputs
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i, unit in enumerate(units):
                if i == len(units) - 1:
                    act = last_act
                output = tf.layers.dense(output, unit, activation=act,
                                        kernel_initializer=tf.glorot_uniform_initializer())
            return output
        
    def forward(self, x, h):
        # 前向传播
        input = tf.concat([x, h], axis=1)
        hs = [self.in_d] + self.hidden_layers + [1]
        output = self.unmm_network(name = "IntegrandNN_forward", inputs = input, units = hs)
        return output + 1.0

class MonotonicNN(object):
    def __init__(self, in_d, hidden_layers, nb_steps=50, device="/cpu:0"):
        # 确保in_d是x_dim + h_dim，其中x_dim=1，h_dim=in_d-1
        self.integrand = IntegrandNN(in_d, hidden_layers)  # in_d应等于1 + h_dim
        self.nb_steps = nb_steps
        self.hidden_layers = hidden_layers
        self.in_d = in_d
        # with tf.variable_scope("MonotonicNN"):
        #     self.net = []
        #     hs = [in_d-1] + hidden_layers + [2]  # h的维度为in_d-1
        #     for h0, h1 in zip(hs, hs[1:]):
        #         self.net.append(tf.layers.Dense(h1, activation=tf.nn.relu))
        #     self.net[-1].activation = None  # 输出层无激活
    
    def unmm_network(self, name, inputs, units, act=tf.nn.relu, last_act=tf.nn.relu, stop_gradient=False):
        if stop_gradient:
            output = tf.stop_gradient(inputs)
        else:
            output = inputs
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i, unit in enumerate(units):
                if i == len(units) - 1:
                    act = last_act
                output = tf.layers.dense(output, unit, activation=act,
                                        kernel_initializer=tf.glorot_uniform_initializer())
            return output
        
    def _parallel_neural_integral(self, x0, x, h):
        # 实现并行神经积分的前向传播
        cc_weights, steps = compute_cc_weights(self.nb_steps)
        xT = x0 + (x - x0) * self.nb_steps / self.nb_steps  # 保持形状
        
        # 扩展维度用于广播
        x0_t = tf.tile(tf.expand_dims(x0, 1), [1, self.nb_steps+1, 1, 1])
        xT_t = tf.tile(tf.expand_dims(xT, 1), [1, self.nb_steps+1, 1, 1])
        h_steps = tf.tile(tf.expand_dims(h, 1), [1, self.nb_steps+1, 1])
        steps_t = tf.tile(tf.expand_dims(steps, 0), [tf.shape(x0)[0], 1, 1])
        
        # 计算积分节点
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2.0
        X_steps_flat = tf.reshape(X_steps, [-1, tf.shape(X_steps)[2]])
        h_steps_flat = tf.reshape(h_steps, [-1, tf.shape(h)[1]])
        
        # 计算被积函数值
        if self.integrand.inv_f:
            dzs = 1.0 / self.integrand.forward(X_steps_flat, h_steps_flat)
        else:
            dzs = self.integrand.forward(X_steps_flat, h_steps_flat)
        
        dzs = tf.reshape(dzs, tf.shape(xT_t))
        dzs_weighted = dzs * tf.expand_dims(cc_weights, 0)
        z_est = tf.reduce_sum(dzs_weighted, axis=1)
        return z_est * (xT - x0) / 2.0

    def forward(self, x, h):
        # 前向计算
        x0 = tf.zeros_like(x)
        with tf.variable_scope("MonotonicNN/forward"):
            # 计算缩放和偏移参数
            input = h
            output = self.unmm_network(name = "MonotonicNN/forward/offandscal", inputs = input ,units = self.hidden_layers)
            # net = h
            # for layer in self.net[:-1]:
            #     net = layer(net)
            #     net = tf.nn.relu(net)
            # out = self.net[-1](net)
            offset = output[:, 0:1]
            scaling = tf.exp(output[:, 1:2])
            
            # 计算积分
            integral = self._parallel_neural_integral(x0, x, h)
            return scaling * integral + offset

def _flatten(tensor_list):
    # 展平张量列表
    return tf.concat([tf.reshape(t, [-1]) for t in tensor_list], axis=0)

def integrate_gradients(x0, x, integrand, h, grad_output, nb_steps, inv_f):
    # 实现梯度计算（反向传播）
    cc_weights, steps = compute_cc_weights(nb_steps)
    x_tot = grad_output * (x - x0) / 2.0
    
    # 扩展维度用于广播
    x0_t = tf.tile(tf.expand_dims(x0, 1), [1, nb_steps+1, 1, 1])
    xT = x
    xT_t = tf.tile(tf.expand_dims(xT, 1), [1, nb_steps+1, 1, 1])
    h_steps = tf.tile(tf.expand_dims(h, 1), [1, nb_steps+1, 1])
    steps_t = tf.tile(tf.expand_dims(steps, 0), [tf.shape(x0)[0], 1, 1])
    
    # 计算积分节点
    X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2.0
    X_steps_flat = tf.reshape(X_steps, [-1, tf.shape(X_steps)[2]])
    h_steps_flat = tf.reshape(h_steps, [-1, tf.shape(h)[1]])
    x_tot_steps = tf.reshape(tf.tile(tf.expand_dims(x_tot, 1), [1, nb_steps+1, 1]) * cc_weights, [-1, tf.shape(x_tot)[1]])
    
    # 计算梯度
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([X_steps_flat, h_steps_flat])
        if inv_f:
            f = 1.0 / integrand.forward(X_steps_flat, h_steps_flat)
        else:
            f = integrand.forward(X_steps_flat, h_steps_flat)
    
    # 参数梯度
    params = tf.trainable_variables(scope="IntegrandNN")
    grads_param = tape.gradient(f, params, output_gradients=x_tot_steps)
    grads_param_flat = _flatten(grads_param)
    
    # h的梯度
    grads_h = tape.gradient(f, h_steps_flat, output_gradients=x_tot_steps)
    grads_h = tf.reshape(grads_h, [tf.shape(x0)[0], nb_steps+1, -1])
    grads_h = tf.reduce_sum(grads_h, axis=1)
    
    return grads_param_flat, grads_h
