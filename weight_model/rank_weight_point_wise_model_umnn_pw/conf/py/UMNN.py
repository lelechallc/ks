import tensorflow as tf
import numpy as np
import math


def _flatten(sequence):
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, axis=0) if len(flat) > 0 else tf.constant([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = tf.constant(lam.T @ W, dtype=tf.float32)
    steps = tf.constant(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps), dtype=tf.float32)

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None, inv_f=False):
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    xT = x0 + nb_steps*step_sizes
    if not compute_grad:
        x0_t = tf.expand_dims(x0, 1)
        x0_t = tf.tile(x0_t, [1, nb_steps + 1, 1])
        xT_t = tf.expand_dims(xT, 1)
        xT_t = tf.tile(xT_t, [1, nb_steps + 1, 1])
        h_steps = tf.expand_dims(h, 1)
        h_steps = tf.tile(h_steps, [1, nb_steps + 1, 1])
        steps_t = tf.expand_dims(steps, 0)
        steps_t = tf.tile(steps_t, [tf.shape(x0_t)[0], 1, tf.shape(x0_t)[2]])
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
        X_steps = tf.reshape(X_steps, [-1, tf.shape(x0_t)[2]])
        h_steps = tf.reshape(h_steps, [-1, tf.shape(h)[1]])
        if inv_f:
            dzs = 1/integrand(X_steps, h_steps)
        else:
            dzs = integrand(X_steps, h_steps)
        dzs = tf.reshape(dzs, [tf.shape(xT_t)[0], nb_steps+1, -1])
        cc_weights_expanded = tf.expand_dims(cc_weights, 0)
        cc_weights_expanded = tf.tile(cc_weights_expanded, [tf.shape(dzs)[0], 1, 1])
        dzs = dzs * cc_weights_expanded
        z_est = tf.reduce_sum(dzs, axis=1)
        return z_est*(xT - x0)/2
    else:
        with tf.GradientTape() as tape:
            x0_t = tf.expand_dims(x0, 1)
            x0_t = tf.tile(x0_t, [1, nb_steps + 1, 1])
            xT_t = tf.expand_dims(xT, 1)
            xT_t = tf.tile(xT_t, [1, nb_steps + 1, 1])
            x_tot = x_tot * (xT - x0) / 2
            x_tot_steps = tf.expand_dims(x_tot, 1)
            x_tot_steps = tf.tile(x_tot_steps, [1, nb_steps + 1, 1])
            cc_weights_expanded = tf.expand_dims(cc_weights, 0)
            cc_weights_expanded = tf.tile(cc_weights_expanded, [tf.shape(x_tot)[0], 1, tf.shape(x_tot)[1]])
            x_tot_steps = x_tot_steps * cc_weights_expanded
            h_steps = tf.expand_dims(h, 1)
            h_steps = tf.tile(h_steps, [1, nb_steps + 1, 1])
            steps_t = tf.expand_dims(steps, 0)
            steps_t = tf.tile(steps_t, [tf.shape(x0_t)[0], 1, tf.shape(x0_t)[2]])
            X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
            X_steps = tf.reshape(X_steps, [-1, tf.shape(x0_t)[2]])
            h_steps = tf.reshape(h_steps, [-1, tf.shape(h)[1]])
            x_tot_steps = tf.reshape(x_tot_steps, [-1, tf.shape(x_tot)[1]])

            g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps+1, inv_f=inv_f)
            return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps, inv_f=False):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(h)
        if inv_f:
            f = 1/integrand(x, h)
        else:
            f = integrand(x, h)

        g_param = _flatten(tape.gradient(f, integrand.trainable_variables, output_gradients=x_tot))
        g_h = _flatten(tape.gradient(f, h, output_gradients=x_tot))

    return g_param, tf.reduce_sum(tf.reshape(g_h, [tf.cast(tf.shape(x)[0]/nb_steps, tf.int32), nb_steps, -1]), axis=1)


class ParallelNeuralIntegral(tf.Module):

    # @tf.function
    def forward(self, x0, x, integrand, flat_params, h, nb_steps=20, inv_f=False):

        x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False, inv_f=inv_f)
        self.integrand = integrand
        self.nb_steps = nb_steps
        self.inv_f = inv_f
        return x_tot

    # @tf.function
    def backward(self, grad_output):
        x0, x, h = self.saved_tensors
        integrand = self.integrand
        nb_steps = self.nb_steps
        inv_f = self.inv_f
        integrand_grad, h_grad = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output, inv_f)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad * grad_output, x_grad * grad_output, None, integrand_grad, tf.reshape(h_grad, h.shape), None