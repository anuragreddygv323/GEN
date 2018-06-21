#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
from __future__ import print_function

import os
import time

import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES']=''

import tensorflow as tf
import numpy as np
import pickle

import reader
import model
import logging
import utils_double

# For train
def run_epoch(session, list_model, verbose=False):
    start_time = time.time()

    costs = [0.0, 0.0, 0.0]
    iters = [0, 0,0]

    for step in range(list_model[0].input.epoch_size):
        for i in range(len(list_model)):
            model = list_model[i]
            fetches = {
                "cost": model.cost,
                "eval_op": model.train_op,
            }
            vals = session.run(fetches)

            cost = vals["cost"]

            costs[i] += cost
            iters[i] += model.input.num_steps

            if verbose and step % (model.input.epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / model.input.epoch_size, np.exp(costs[i] / iters[i]),
                       iters[i] * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs[0] / iters[0])

# For Valid
def run_epoch2(session, model):
    costs = 0.0
    iters = 0.0

    fetches = {
        "cost": model.cost,
    }
    # 每轮每步都会获取num_Step的input数据
    for step in range(model.input.epoch_size):
        vals = session.run(fetches)
        cost = vals["cost"]

        costs += cost
        iters += model.input.num_steps
    return np.exp(costs / iters)


def run_test(session, model, config, mode, feed_dict=None, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()

    fetches = {
        "cost": model.cost,
        "predict1": model.predict1,  # 预测的softmax向量
        "predict2": model.predict2,  # 预测的softmax向量
        "predict3": model.predict3,  # 预测的softmax向量
        "triples": model.triples  # 给定三元组
    }

    costs = 0.0
    iters = 0

    results1 = {
        "triples": [],  # 给定三元组
        "softmax": []  # 预测的softmax向量
    }

    results2 = {
        "triples": [],  # 给定三元组
        "softmax": []  # 预测的softmax向量
    }

    results3 = {
        "triples": [],  # 给定三元组
        "softmax": []  # 预测的softmax向量
    }

    for step in range(model.input.epoch_size):
        vals = session.run(fetches, feed_dict)

        pred1 = vals["predict1"]
        pred2 = vals["predict2"]
        pred3 = vals["predict3"]
        fact = vals["triples"]
        results1["triples"].append(fact)
        results2["triples"].append(fact)
        results3["triples"].append(fact)
        results1["softmax"].append(pred1)
        results2["softmax"].append(pred2)
        results3["softmax"].append(pred3)

        cost = vals["cost"]
        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.4f perplexity: %.4f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    ISOTIMEFORMAT = '%m%d%H%M'
    timestr = datetime.datetime.now()

    perplexity = np.exp(costs / iters)

    return (perplexity, results1, results2, results3)


def main(_):
    log_dir = "./results/"
    utils_double.set_logger(log_dir)
    base_dir = "./data/WN18/"
    starttime = time.time()
    config = utils_double.get_config("small")

    eval_config = utils_double.get_config("small")
    eval_config.batch_size = 1
    eval_config.num_steps = 3

    if not config.data_path:
        raise ValueError("Must set --data_path to data directory")

    logging.info('traing')
    logging.info('init_scale:%f' % config.init_scale)
    logging.info('learning_rate:%f' % config.learning_rate)
    logging.info('max_grad_norm:%f' % config.max_grad_norm)
    logging.info('num_layer s:%f' % config.num_layers)
    logging.info('num_steps:%f' % config.num_steps)
    logging.info('hidden_size:%f' % config.hidden_size)
    logging.info('max_epoch:%f' % config.max_epoch)
    logging.info('max_max_epoch:%f' % config.max_max_epoch)
    # logging.info('keep_prob:%f' % config.keep_prob)
    logging.info('lr_decay:%f' % config.lr_decay)
    logging.info('batch_size:%f' % config.batch_size)
    logging.info('vocab_size:%f' % config.vocab_size)
    logging.info('mlp_dim:%f' % config.mlp_dim)

    train_data, test_data,valid_data, nvocab = reader.raw_data(config.data_path)
    print(nvocab)
    print(config.nent)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        train_input = model.Producer(config=config, data=train_data, is_traing=True, name="TrainInput")
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model1 = model.GEN(is_training=True, config=config, input_=train_input,task = [1,0,0])

        with tf.name_scope("Train"):
            train_input = model.Producer(config=config, data=train_data, is_traing=True, name="TrainInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                train_model2 = model.GEN(is_training=True, config=config, input_=train_input,task = [0,1,0])

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                train_model3 = model.GEN(is_training=True, config=config, input_=train_input,task = [0,0,1])

        with tf.name_scope("Test"):
            test_input = model.Producer(config=eval_config, data=test_data, is_traing=False, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = model.GEN(is_training=False, config=eval_config, input_=test_input,task = [1,1,1])


        # with tf.name_scope("valid"):
        #     valid_input = model.Producer(config=eval_config, data=valid_data, is_traing=False, name="ValidInput")
        #     with tf.variable_scope("Model", reuse=True,initializer=initializer):
        #         valid_model = model.GEN(is_training=False, config=eval_config,input_=valid_input, task=[1, 1, 1])

        sv = tf.train.Supervisor(logdir=config.save_path, save_model_secs=1000000)

        skip = False
        with sv.managed_session() as session:

            # state = session.run(train_model.initial_state)

            for i in range(config.max_max_epoch):  # 训练多少轮
                if skip == False:
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    cand = config.learning_rate * lr_decay
                    if (cand < 0.0000001):
                        lr = 0.0000001
                        skip = True
                    else:
                        lr = cand
                    train_model1.assign_lr(session, lr)
                    train_model2.assign_lr(session, lr)
                    train_model3.assign_lr(session, lr)
                print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(train_model1.lr)))


                if i < 10:
                    list_model = [train_model1,train_model2,train_model3]
                elif i < 20:
                    list_model = [train_model1,train_model3]
                else:
                    list_model = [train_model1,train_model2,train_model3]


                train_perplexity = run_epoch(session, list_model, verbose=True)  # 每轮的训练

                print("Epoch: %d Train Perplexity: %.6f" % (i + 1, train_perplexity))

                # valid_perplexity = run_epoch2(session, valid_model)
                # print("Epoch: %d Valid Perplexity: %.6f" % (i + 1, valid_perplexity))
                logging.info("%.4f" % (train_perplexity))

            if config.save_path:
                print("Saving model to %s." % config.save_path)
                sv.saver.save(session, config.save_path, global_step=sv.global_step)

            ISOTIMEFORMAT = '%m%d%H%M'
            timestr = datetime.datetime.now()
            fname = config.out_path + "emb_gen_" + str(timestr.strftime(ISOTIMEFORMAT)) + ".npy"
            np.save(fname, session.run(train_model1.embedding))

            test_perplexity, res1, res2, res3 = run_test(session, test_model, eval_config, 1)
            print("Test Perplexity: %.4f" % test_perplexity)


    #
    utils_double.calc_acc(config, base_dir, res1, 1)
    utils_double.calc_acc(config, base_dir, res2, 2)
    utils_double.calc_acc(config, base_dir, res3, 3)

    endtime = time.time()

    print("use time：%s" % (endtime - starttime))
    print("---- we are all done! ----")


if __name__ == "__main__":
    tf.app.run()





