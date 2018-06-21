# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import numpy as np


def raw_data(data_path=None):
    """
    Load raw data from data directory "data_path".
    Reads text files, converts strings to integer ids,
    and performs mini-batching of the inputs.caption
    Args:
        data_path: string path to the directory where simple-examples.tgz has been extracted.
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
    """

    train_path = os.path.join(data_path, "train_id.txt")
    test_path = os.path.join(data_path, "test_id.txt")
    valid_path = os.path.join(data_path, "valid_id.txt")

    with tf.gfile.GFile(train_path, "r") as f:
        line = f.read()
        rep = line.replace("\n", "\t")
        data = rep.split()
        train_data = [word for word in data]

        counter = collections.Counter(train_data)
        vocabulary = len(counter)

    with tf.gfile.GFile(test_path, "r") as f:
        line = f.read()
        rep = line.replace("\n", "\t")
        data = rep.split()
        test_data = [word for word in data]

    with tf.gfile.GFile(valid_path, "r") as f:
        line = f.read()
        rep = line.replace("\n", "\t")
        data = rep.split()
        valid_data = [word for word in data]

    return train_data, test_data, valid_data, vocabulary



def data_producer(raw_data, batch_size, num_steps, name=None, shuffle=None):
    """
    Iterate on the raw data.
    This chunks up raw_data into batches of examples and returns Tensors that are drawn from these batches.
    Args:
        raw_data: one of the raw data outputs from raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).
    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps].
        The second element of the tuple is the same 数据集 time-shifted to the right by one.
    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "Producer", [raw_data, batch_size, num_steps]):
        raw_data = np.array(raw_data)
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)

        batch_len = data_len // batch_size  # batch_size表示每批次数据的个数
        batch_len = (batch_len // num_steps) * num_steps

        # 数据的shape被重新组织为：batch_size * batch_len
        data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps

        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        # 产生一共序列，序列包含生成0～epoch_size-1个元素，如果num_epochs有指定，则每个元素只产生num_epochs次，否则循环产生
        i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
        # 取数据
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])

        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])


    return x, y

