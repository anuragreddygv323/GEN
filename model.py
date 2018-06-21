#!/usr/bin/env python
# encoding: utf-8

"""
Example for building a LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf

import reader
import numpy as np


class Producer(object):
    """The input data."""

    def __init__(self, config, data, is_traing, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps

        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        if (is_traing):
            self.input_data, self.targets = reader.data_producer(data, batch_size, num_steps, name=name, shuffle=True)
        else:
            self.input_data, self.targets = reader.data_producer(data, batch_size, num_steps, name=name, shuffle=False)


# 同10,只是使用2个mlp_w,2个softmax,一个embedding,并根据任务更新参数
class GEN(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_, task):
        self._input = input_

        self.task = task

        batch_size = input_.batch_size
        vocab_size = config.vocab_size

        size = config.hidden_size * 2
        nent = config.nent


        # 获得输入数据

        self._embedding = tf.get_variable("embedding", [vocab_size, config.hidden_size], dtype=tf.float32)

        inputs = tf.nn.embedding_lookup(self._embedding, input_.input_data)

        loss = loss1 = loss2 = loss3 = 0

        self.mlp_we = tf.get_variable("mlp_w1", [size, config.mlp_dim], dtype=tf.float32)
        self.mlp_be = tf.get_variable("mlp_b1", [config.mlp_dim], dtype=tf.float32)

        # self.mlp_wr = tf.get_variable("mlp_w2", [size, config.mlp_dim], dtype=tf.float32)
        self.mlp_wr = tf.get_variable("mlp_w2", [size, config.mlp_dim_r], dtype=tf.float32)
        self.mlp_br = tf.get_variable("mlp_b2", [config.mlp_dim_r], dtype=tf.float32)

        self.softmax_we = tf.get_variable("softmax_w1", [config.mlp_dim, nent], dtype=tf.float32,
                                          trainable=is_training)
        self.softmax_be = tf.get_variable("softmax_b1", [nent], dtype=tf.float32, trainable=is_training)

        self.softmax_wr = tf.get_variable("softmax_w2", [config.mlp_dim_r, config.nrel], dtype=tf.float32,
                                          trainable=is_training)
        self.softmax_br = tf.get_variable("softmax_b2", [config.nrel], dtype=tf.float32, trainable=is_training)

        if task[2] == 1:
            # concat h and r to predict t
            inputs_mlp = tf.concat([inputs[:, 0, :], inputs[:, 1, :]], axis=1)
            inputs_mlp = tf.reshape(inputs_mlp, [batch_size, size])

            mlp_logit = tf.matmul(inputs_mlp, self.mlp_we) + self.mlp_be
            mlp_logit = tf.nn.relu(mlp_logit)
            mlp_logit_drop = tf.nn.dropout(mlp_logit, config.mlp_prob)

            self._logits1 = tf.matmul(mlp_logit_drop, self.softmax_we) + self.softmax_be

            self._logits = self._logits1

            # target = tf.strided_slice(input_.targets, [0, 1], [batch_size, 2])  # r的目标 t
            target = input_.input_data[:, 2]
            target = tf.reshape(target, [batch_size])

            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=self._logits1)
            loss = loss1

        if task[0] == 1:
            # caoncat t and r to predict h
            inputs_mlp = tf.concat([inputs[:, 2, :], inputs[:, 1, :]], axis=1)
            inputs_mlp = tf.reshape(inputs_mlp, [batch_size, size])

            mlp_logit = tf.matmul(inputs_mlp, self.mlp_we) + self.mlp_be
            mlp_logit = tf.nn.relu(mlp_logit)
            mlp_logit_drop = tf.nn.dropout(mlp_logit, config.mlp_prob)

            self._logits2 = tf.matmul(mlp_logit_drop, self.softmax_we) + self.softmax_be

            self._logits = self._logits2

            target = input_.input_data[:, 0]
            target = tf.reshape(target, [batch_size])
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=self._logits2)
            loss = loss2

        if task[1] == 1:
            inputs_mlp = tf.concat([inputs[:, 0, :], inputs[:, 2, :]], axis=1)
            inputs_mlp = tf.reshape(inputs_mlp, [batch_size, size])

            mlp_logit = tf.matmul(inputs_mlp, self.mlp_wr) + self.mlp_br
            mlp_logit = tf.nn.relu(mlp_logit)
            mlp_logit_drop = tf.nn.dropout(mlp_logit, config.mlp_prob)

            self._logits3 = tf.matmul(mlp_logit_drop, self.softmax_wr) + self.softmax_br

            self._logits = self._logits3

            a = tf.convert_to_tensor(np.array([[config.nent]] * batch_size), dtype=tf.int32)
            b = tf.reshape(input_.input_data[:, 1], [batch_size, 1])
            target = tf.subtract(b, a)
            target = tf.reshape(target, [batch_size])

            loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=self._logits3)
            loss = loss3


        self._cost = cost = tf.reduce_mean(loss)  # 取平均误差

        if not is_training:
            self._predict = tf.nn.softmax(self._logits)
            self._predict1 = tf.nn.softmax(self._logits2)
            self._predict2 = tf.nn.softmax(self._logits3)
            self._predict3 = tf.nn.softmax(self._logits1)

            self._h = tf.reshape(input_.input_data[:, 0], [batch_size])
            self._r = tf.reshape(input_.input_data[:, 1], [batch_size])
            self._t = tf.reshape(input_.input_data[:, 2], [batch_size])
            return

        self._lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()


        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict(self):
        return self._predict

    @property
    def triples(self):
        return (self._h, self._r, self._t)

    @property
    def embedding(self):
        return self._embedding

    @property
    def predict1(self):
        return self._predict1

    @property
    def predict2(self):
        return self._predict2

    @property
    def predict3(self):
        return self._predict3
