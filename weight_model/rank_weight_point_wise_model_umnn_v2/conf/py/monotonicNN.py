import tensorflow as tf

class IntegrandNN:
    def __init__(self, in_dim, hidden_layers):
        """ 被积函数神经网络 """
        self.layers = []
        # 构建隐藏层
        prev_dim = in_dim
        for h_dim in hidden_layers:
            self.layers.append(tf.keras.layers.Dense(h_dim, activation='relu'))
            prev_dim = h_dim
        # 输出层（无激活函数）
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
        self.elu = tf.keras.layers.ELU()

    def __call__(self, inputs):
        """ 前向传播 """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        # ELU激活后加1保证输出非负
        return self.elu(x) + 1.0

class MonotonicNN:
    def __init__(self, in_dim, hidden_layers, nb_steps=50, device="cpu"):
        """
        单调神经网络
        :param in_dim: 输入维度 (x_dim + h_dim)
        :param hidden_layers: 隐藏层结构
        :param nb_steps: 积分步数
        :param device: 计算设备
        """
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.nb_steps = nb_steps
        self.device = device

        # 被积函数网络（输入x和h拼接）
        self.integrand = IntegrandNN(in_dim, hidden_layers)

        # 条件网络（输入h，输出offset和log_scaling）
        self.condition_net = []
        prev_dim = in_dim - 1  # h的维度为in_dim-1
        for h_dim in hidden_layers:
            self.condition_net.append(tf.keras.layers.Dense(h_dim, activation='relu'))
            prev_dim = h_dim
        self.condition_net.append(tf.keras.layers.Dense(2, activation=None))  # 输出offset和log_scaling

    def __call__(self, x, h):
        """ 前向计算 """
        with tf.device(self.device):
            # 条件网络计算offset和scaling
            cond_out = h
            for layer in self.condition_net:
                cond_out = layer(cond_out)
            offset = cond_out[:, 0:1]
            log_scaling = cond_out[:, 1:2]
            scaling = tf.exp(log_scaling)  # 保证scaling为正

            # 生成积分区间 [0, x] 分为nb_steps步
            x0 = tf.zeros_like(x)  # 积分下限（全0）
            delta_x = (x - x0) / self.nb_steps  # 每步间隔 (batch_size, 1)

            # 生成积分点 (batch_size, nb_steps+1, 1)
            steps = tf.linspace(0.0, 1.0, self.nb_steps + 1)  # (nb_steps+1,)
            steps = tf.reshape(steps, [1, -1, 1])  # (1, nb_steps+1, 1)
            x_steps = x0 + steps * (x - x0)  # 广播计算所有点的位置

            # 扩展h以匹配积分点维度 (batch_size, nb_steps+1, h_dim)
            h_expanded = tf.expand_dims(h, axis=1)
            h_expanded = tf.tile(h_expanded, [1, self.nb_steps+1, 1])

            # 拼接x和h作为被积函数输入 (batch_size, nb_steps+1, in_dim)
            integrand_input = tf.concat([x_steps, h_expanded], axis=-1)
            batch_size = tf.shape(integrand_input)[0]
            integrand_input = tf.reshape(integrand_input, [-1, self.in_dim])  # 展平以输入网络

            # 计算被积函数值 (batch_size*(nb_steps+1), 1)
            f = self.integrand(integrand_input)
            f = tf.reshape(f, [batch_size, self.nb_steps+1, 1])  # 恢复形状

            # 梯形法则计算积分（首尾系数0.5，中间1.0）
            weights = tf.concat([
                tf.constant([0.5], shape=[1, 1]),
                tf.ones([self.nb_steps-1, 1]),
                tf.constant([0.5], shape=[1, 1])
            ], axis=0)
            weights = tf.reshape(weights, [1, -1, 1])  # 广播权重

            # 计算加权和并乘以步长
            weighted_sum = tf.reduce_sum(f * weights, axis=1)  # (batch_size, 1)
            integral = weighted_sum * delta_x  # 积分结果

            # 最终输出
            return scaling * integral + offset
