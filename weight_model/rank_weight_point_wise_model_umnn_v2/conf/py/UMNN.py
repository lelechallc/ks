import tensorflow as tf
import numpy as np
import math

def _flatten(sequence):
    """将张量列表展平为单个张量"""
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.constant([], dtype=tf.float32)

def compute_cc_weights(nb_steps):
    """计算 Clenshaw-Curtis 积分权重和节点"""
    lam = tf.range(0, nb_steps + 1, dtype=tf.float32)
    lam = tf.reshape(lam, [-1, 1])
    lam = tf.cos(tf.matmul(lam, lam, transpose_b=True) * math.pi / nb_steps)
    
    # 调整权重矩阵边界
    mask = tf.concat([tf.ones([nb_steps+1, 1], dtype=tf.bool)[:, 0:1], 
                      tf.tile(False, [nb_steps+1, nb_steps-1]),
                      tf.ones([nb_steps+1, 1], dtype=tf.bool)[:, 0:1]], axis=1)
    lam = tf.where(mask, 0.5 * lam, lam)
    lam = lam * 2.0 / nb_steps
    
    # 计算权重系数
    W = tf.range(0, nb_steps + 1, dtype=tf.float32)
    W = tf.reshape(W, [-1, 1])
    W = tf.where(tf.equal(tf.mod(W, 2), 0), 2.0 / (1 - W**2), tf.zeros_like(W))
    W = tf.where(tf.equal(W, 0), 1.0, W)
    
    # 最终权重计算
    cc_weights = tf.matmul(lam, W, transpose_a=True)
    steps = tf.cos(tf.range(0, nb_steps + 1, dtype=tf.float32) * math.pi / nb_steps)
    return cc_weights, tf.reshape(steps, [-1, 1])

def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None, inv_f=False):
    """执行 Clenshaw-Curtis 积分计算"""
    cc_weights, steps = compute_cc_weights(nb_steps)
    batch_size = tf.shape(x0)[0]
    dim = x0.get_shape().as_list()[-1]
    
    xT = x0 + nb_steps * step_sizes
    
    if not compute_grad:
        # 前向传播计算
        x0_t = tf.tile(tf.expand_dims(x0, 1), [1, nb_steps+1, 1])
        xT_t = tf.tile(tf.expand_dims(xT, 1), [1, nb_steps+1, 1])
        h_steps = tf.tile(tf.expand_dims(h, 1), [1, nb_steps+1, 1])
        steps_t = tf.tile(tf.reshape(steps, [1, -1, 1]), [batch_size, 1, dim])
        
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2.0
        X_steps_flat = tf.reshape(X_steps, [-1, dim])
        h_steps_flat = tf.reshape(h_steps, [-1, h.get_shape().as_list()[-1]])
        
        if inv_f:
            dzs = 1.0 / integrand(X_steps_flat, h_steps_flat)
        else:
            dzs = integrand(X_steps_flat, h_steps_flat)
            
        dzs = tf.reshape(dzs, [batch_size, nb_steps+1, -1])
        weights = tf.tile(tf.expand_dims(cc_weights, 0), [batch_size, 1, 1])
        z_est = tf.reduce_sum(dzs * weights, axis=1)
        return z_est * (xT - x0) / 2.0
    
    else:
        # 反向传播梯度计算
        x_tot = x_tot * (xT - x0) / 2.0
        x_tot_steps = tf.tile(tf.expand_dims(x_tot, 1), [1, nb_steps+1, 1]) * cc_weights
        x_tot_steps_flat = tf.reshape(x_tot_steps, [-1, x_tot.get_shape().as_list()[-1]])
        
        x0_t = tf.tile(tf.expand_dims(x0, 1), [1, nb_steps+1, 1])
        xT_t = tf.tile(tf.expand_dims(xT, 1), [1, nb_steps+1, 1])
        h_steps = tf.tile(tf.expand_dims(h, 1), [1, nb_steps+1, 1])
        steps_t = tf.tile(tf.reshape(steps, [1, -1, 1]), [batch_size, 1, dim])
        
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2.0
        X_steps_flat = tf.reshape(X_steps, [-1, dim])
        h_steps_flat = tf.reshape(h_steps, [-1, h.get_shape().as_list()[-1]])
        
        # 计算积分函数的梯度
        with tf.GradientTape(persistent=True) as g:
            g.watch([X_steps_flat, h_steps_flat])
            if inv_f:
                f = 1.0 / integrand(X_steps_flat, h_steps_flat)
            else:
                f = integrand(X_steps_flat, h_steps_flat)
                
        # 参数梯度
        params = integrand.trainable_variables
        grads_param = g.gradient(f, params, output_gradients=x_tot_steps_flat)
        grads_param_flat = _flatten(grads_param)
        
        # h 的梯度
        grads_h = g.gradient(f, h_steps_flat, output_gradients=x_tot_steps_flat)
        grads_h = tf.reshape(grads_h, [batch_size, nb_steps+1, -1])
        grads_h = tf.reduce_sum(grads_h, axis=1)
        
        return grads_param_flat, grads_h

def parallel_neural_integral(x0, x, integrand, h, nb_steps=20, inv_f=False):
    """自定义积分操作，实现前向和反向传播"""
    # 前向计算
    xT = x0 + (x - x0) * nb_steps / nb_steps  # 实际 step_sizes 为 (x-x0)/nb_steps
    with tf.name_scope('Forward'):
        z = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False, None, inv_f)
    
    # 定义自定义梯度
    @tf.custom_gradient
    def wrapped_integral(x0, x, h):
        def grad_fn(*grad_outputs):
            with tf.name_scope('Backward'):
                grad_output = grad_outputs[0]
                # 计算参数和输入的梯度
                grad_param, grad_h = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output, inv_f)
                
                # 计算 x0 和 x 的梯度 (Leibniz 公式)
                grad_x0 = -integrand(x0, h) * grad_output
                grad_x = integrand(x, h) * grad_output
                
                return grad_x0, grad_x, grad_h
        
        return z, grad_fn
    
    return wrapped_integral(x0, x, h)