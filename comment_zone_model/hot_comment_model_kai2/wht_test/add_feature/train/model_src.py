""" 似乎不能把计算图封装到class，Kai会报错。还是用函数式编程吧.
"""
import tensorflow.compat.v1 as tf

class ExplicitCrossModel:
	def __init__(self, last_dim, layer_num, projection_dim=None, final_proj_dim=None):
		self.last_dim = last_dim
		self.layer_num = layer_num
		self.projection_dim = projection_dim	# 一般projection_dim<<last_dim
		self.final_proj_dim = final_proj_dim

		self._kernel_initializer = tf.keras.initializers.get("truncated_normal")
		self._bias_initializer = tf.keras.initializers.get("zeros")
		self._kernel_regularizer = tf.keras.regularizers.get(None)
		self._bias_regularizer = tf.keras.regularizers.get(None)

		self._dense_kernels, self._dense_u_kernels, self._dense_v_kernels = [], [], []
		for _ in range(self.layer_num):
			if self.projection_dim is None:
				self._dense_kernels.append(tf.keras.layers.Dense(
					last_dim,
					activation='relu',
					kernel_initializer=self._clone_initializer(
						self._kernel_initializer),
					bias_initializer=self._bias_initializer,
					kernel_regularizer=self._kernel_regularizer,
					bias_regularizer=self._bias_regularizer,
					use_bias=True,
				))
			else:
				self._dense_u_kernels.append(tf.keras.layers.Dense(
					self.projection_dim,
					kernel_initializer=self._clone_initializer(
						self._kernel_initializer),
					kernel_regularizer=self._kernel_regularizer,
					use_bias=False,
				))
				self._dense_v_kernels.append(tf.keras.layers.Dense(
					last_dim,
					activation='relu',
					kernel_initializer=self._clone_initializer(
						self._kernel_initializer),
					bias_initializer=self._bias_initializer,
					kernel_regularizer=self._kernel_regularizer,
					bias_regularizer=self._bias_regularizer,
					use_bias=True,
				))
		
		if self.final_proj_dim is not None:
			self._final_dense = tf.keras.layers.Dense(
				self.final_proj_dim,
				activation='relu',
				kernel_initializer=self._clone_initializer(
					self._kernel_initializer),
				bias_initializer=self._bias_initializer,
				kernel_regularizer=self._kernel_regularizer,
				bias_regularizer=self._bias_regularizer,
				use_bias=True,
			)

	def forward(self, x0):
		xl = x0
		for i in range(self.layer_num):
			if self.projection_dim is None:
				prod_output = self._dense_kernels[i](xl)
			else:
				prod_output = self._dense_v_kernels[i](self._dense_u_kernels[i](xl))
			xl = x0 * prod_output + xl
		if self.final_proj_dim is not None:
			xl = self._final_dense(xl)
		return xl

	# def _clone_initializer(self, initializer):
	# 	return initializer.__class__.from_config(initializer.get_config())


class TowerModel:
	def __init__(self, name, units) -> None:
		self.name = name
		self.units = units
		self._sequential = tf.keras.Sequential()
		for i, unit in enumerate(units):
			if i == len(units)-1:
				act = 'sigmoid'
			else:
				act = 'relu'
			self._sequential.add(tf.keras.layers.Dense(unit, activation=act))
	
	def forward(self, inputs):
		output = self._sequential(inputs)
		return output


class Model:
	def __init__(self, input_dim, tower_names=['expand', 'like', 'reply'], tower_units=[256, 128, 64, 1], 
				explicit_cross_layer_num=None, explicit_cross_projection_dim=None, explicit_cross_final_proj_dim=None,
				implicit_cross_layer_units=[], projection_dim_for_extra_input=None, dropout=0):
		self.tower_names = tower_names	# 模型的preds按照传入的tower顺序返回
		self.dropout = dropout		# 当前dropout未生效！

		self.implicit_cross_module = [tf.keras.layers.Dense(
			units, activation='relu') for units in implicit_cross_layer_units]
		
		if explicit_cross_layer_num is not None:
			self.explicit_cross_module = ExplicitCrossModel(input_dim, explicit_cross_layer_num, explicit_cross_projection_dim, explicit_cross_final_proj_dim)
		else:
			self.explicit_cross_module = None
		
		if projection_dim_for_extra_input is not None:
			self.projection_layer_for_extra_input = tf.keras.layers.Dense(projection_dim_for_extra_input, activation='relu')
		else:
			self.projection_layer_for_extra_input = None
		self.tower_module = {name: TowerModel(name, tower_units) for name in self.tower_names}


	def forward(self, inputs, extra_inputs=None):
		implicit_cross_output = inputs
		for layer in self.implicit_cross_module:
			implicit_cross_output = layer(implicit_cross_output)
		
		bottom_outputs = [implicit_cross_output]
		if self.explicit_cross_module is not None:
			explicit_cross_output = self.explicit_cross_module.forward(inputs)
			bottom_outputs.append(explicit_cross_output)

		if extra_inputs is not None:
			if self.projection_layer_for_extra_input is None:
				bottom_outputs.append(extra_inputs)
			else:
				extra_inputs = self.projection_layer_for_extra_input(extra_inputs)
				bottom_outputs.append(extra_inputs)

		bottom_output = tf.concat(bottom_outputs, -1)
		outputs = []
		for name in self.tower_names:
			outputs.append(self.tower_module[name].forward(bottom_output))
		
		return outputs
