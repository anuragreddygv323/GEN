#!/usr/bin/env python
# encoding: utf-8

import numpy as np

import pickle
import datetime
import logging

import heapq
from sklearn.metrics import f1_score


class SmallConfig(object):
    """Small config."""
    data_path = "./data/WN18"
    save_path = "./models/"
    out_path = "./results/"

    nent = 40943  # wn18
    nrel = 18


    model = "small"
    init_scale = 0.1
    learning_rate = 3

    max_grad_norm = 10

    num_layers = 1
    num_steps = 3
    hidden_size = 100
    max_epoch = 6
    max_max_epoch = 50
    keep_prob = 0.6
    lr_decay = 0.9
    batch_size = 256
    vocab_size = 40961
    mlp_dim = 2048
    mlp_dim_r = 512
    mlp_prob = 0.8


class MediumConfig(object):
    """Small config."""
    data_path = "./data/WN18"
    save_path = "./models/"
    out_path = "./results/"

    nent = 40943  # wn18
    nrel = 18

    model = "medium"
    init_scale = 0.1
    learning_rate = 3

    max_grad_norm = 10

    num_layers = 1
    num_steps = 3
    hidden_size = 100
    max_epoch = 6
    max_max_epoch = 50
    keep_prob = 0.6
    lr_decay = 0.9
    batch_size = 256
    vocab_size = 40961
    mlp_dim = 2048
    mlp_dim_r = 512
    mlp_prob = 0.8


class LargeConfig(object):
    """Small config."""
    data_path = "./data/WN18"
    save_path = "./models/"
    out_path = "./results/"

    nent = 40943  # wn18
    nrel = 18

    model = "large"
    init_scale = 0.1
    learning_rate = 3

    max_grad_norm = 10

    num_layers = 1
    num_steps = 3
    hidden_size = 100
    max_epoch = 6
    max_max_epoch = 50
    keep_prob = 0.6
    lr_decay = 0.9
    batch_size = 256
    vocab_size = 40961
    mlp_dim = 2048
    mlp_dim_r = 512
    mlp_prob = 0.8

class TestConfig(object):
    """Small config."""
    data_path = "./data/WN18"
    save_path = "./models/"
    out_path = "./results/"

    nent = 40943  # wn18
    nrel = 18

    model = "test"
    init_scale = 0.1
    learning_rate = 3

    max_grad_norm = 10

    num_layers = 1
    num_steps = 3
    hidden_size = 100
    max_epoch = 6
    max_max_epoch = 50
    keep_prob = 0.6
    lr_decay = 0.9
    batch_size = 256
    vocab_size = 40961
    mlp_dim = 2048
    mlp_dim_r = 512
    mlp_prob = 0.8

def get_config(model_type):
    if model_type == "small":
        config = SmallConfig()
    elif model_type == "medium":
        config = MediumConfig()
    elif model_type == "large":
        config = LargeConfig()
    elif model_type == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", model_type)

    return config


def set_logger(log_dir, mode=None):
    ISOTIMEFORMAT = '%m%d%H%M%S'
    timestr = datetime.datetime.now()
    fname = log_dir + str(timestr.strftime(ISOTIMEFORMAT)) + ".log"

    if mode is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            filename=fname,
                            filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%b %d %H:%M:%S',
                            filename=fname,
                            filemode='w')

def read_triples(base_dir, fname):

    hlist = []
    rlist = []
    tlist = []

    hset = {}  # 给定(t,r)，已知的h集合
    tset = {}  # 给定(h,r)，已知的t集合
    rset = {}  # 给定(h,t)，已知的r集合

    # 首先读入训练数据
    fdat = base_dir + "base/train_id.txt"
    with open(fdat) as f:
        if fname == "train":
            for line in f.readlines():
                (h, r, t) = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                hlist.append(h)
                rlist.append(r)
                tlist.append(t)

                hset.setdefault(r, {})
                hset[r].setdefault(t, set()).add(h)
                tset.setdefault(r, {})
                tset[r].setdefault(h, set()).add(t)
                rset.setdefault(h, {})
                rset[h].setdefault(t, set()).add(r)
        else:
            for line in f.readlines():
                (h, r, t) = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                hset.setdefault(r, {})
                hset[r].setdefault(t, set()).add(h)
                tset.setdefault(r, {})
                tset[r].setdefault(h, set()).add(t)
                rset.setdefault(h, {})
                rset[h].setdefault(t, set()).add(r)

    if fname == "test":
        fdat = base_dir + "base/test_id.txt"  # 正向数据
        with open(fdat) as f:
            for line in f.readlines():
                (h, r, t) = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                hlist.append(h)
                rlist.append(r)
                tlist.append(t)

                hset.setdefault(r, {})
                hset[r].setdefault(t, set()).add(h)
                tset.setdefault(r, {})
                tset[r].setdefault(h, set()).add(t)
                rset.setdefault(h, {})
                rset[h].setdefault(t, set()).add(r)

        fdat = base_dir + "base/valid_id.txt"  # 正向数据
        with open(fdat) as f:
            for line in f.readlines():
                (h, r, t) = line.strip().split()
                h = int(h)
                r = int(r)
                t = int(t)
                hset.setdefault(r, {})
                hset[r].setdefault(t, set()).add(h)
                tset.setdefault(r, {})
                tset[r].setdefault(h, set()).add(t)
                rset.setdefault(h, {})
                rset[h].setdefault(t, set()).add(r)

    dataset = {"hlist": hlist,
               "rlist": rlist,
               "tlist": tlist,
               "hset": hset,
               "tset": tset,
               "rset": rset}

    return dataset


def read_data(base_dir, fname):
    dataset = read_triples(base_dir, fname)

    print("number of triples = " + str(len(dataset["hlist"])))

    return dataset


def find_rank(base_dir, target, pred_list, nsize, tset):
    rank = 0
    rank_filter = 0
    for i in range(nsize):
        if (target < pred_list[i]):
            rank += 1
            if (i in tset):
                pass
                # print("bingo!")
            else:
                rank_filter += 1
                #     print("rank = %d  filtered = %d" % (rank, rank_filter) )
    # print("%d,  %d" % (rank, rank_filter))

    return (rank, rank_filter)


# 用于3种任务的测试计算,and calculate the p@1  p@3 p@5 p@10
def calc_acc(config, base_dir, results, mode):  # mode  == 0,1,2  表示预测h t r
    print("calculating prediction accuracy ...")
    triples = results["triples"]
    predict = results["softmax"]

    ndata = len(triples)
    dataset = read_data(base_dir, 'test')

    nent = config.nent
    nrel = config.nrel


    tset = dataset["tset"]
    hset = dataset["hset"]
    rset = dataset["rset"]
    p_10 = 0.0
    p_1 = 0.0
    p_3 = p_5 = p_7 = 0.0
    p_10_filter = 0.0
    p_1_filter = 0.0
    p_3_filter = p_5_filter = p_7_filter = 0.0
    ranklist = np.zeros(ndata)
    ranklist_filter = np.zeros(ndata)

    output = []
    if (mode == 1):
        output.append('r t h \n')
    elif (mode == 2):
        output.append('h t r \n')
    elif (mode == 3):
        output.append('h r t \n')

    for i in range(ndata):
        # if(mode == 1):
        h = int(triples[i][0])
        r = int(triples[i][1])
        r_base = int(triples[i][1]) - nent
        t = int(triples[i][2])
        if (mode == 1):  # predict head
            prob = predict[i][0][h]
            (rank, rank_filter) = find_rank(base_dir, prob, predict[i][0], nent, hset[r_base][t])
            output.append(
                '[' + str(h) + '\t' + str(r) + '\t' + str(t) + ']\t' + str(rank) + '\t' + str(rank_filter) + '\n')
        elif (mode == 2):  # predict rel
            prob = predict[i][0][r_base]
            (rank, rank_filter) = find_rank(base_dir, prob, predict[i][0], nrel, rset[h][t])
            output.append(
                '[' + str(h) + '\t' + str(r) + '\t' + str(t) + ']\t' + str(rank) + '\t' + str(rank_filter) + '\n')
        elif (mode == 3):  # predict tail
            prob = predict[i][0][t]
            (rank, rank_filter) = find_rank(base_dir, prob, predict[i][0], nent, tset[r_base][h])
            output.append(
                '[' + str(h) + '\t' + str(r) + '\t' + str(t) + ']\t' + str(rank) + '\t' + str(rank_filter) + '\n')


        ranklist[i] = rank
        if rank < 10:
            p_10 += 1
            if rank < 7:
                p_7 += 1
                if rank < 5:
                    p_5 += 1
                    if rank < 3:
                        p_3 += 1
                        if rank < 1:
                            p_1 += 1

        ranklist_filter[i] = rank_filter
        if rank_filter < 10:
            p_10_filter += 1
            if rank_filter < 7:
                p_7_filter += 1
                if rank_filter < 5:
                    p_5_filter += 1
                    if rank_filter < 3:
                        p_3_filter += 1
                        if rank_filter < 1:
                            p_1_filter += 1

    output.append("-" * 60 + "\n")
    if (config.forward == 1):
        output.append("RNN model (replace tails -- right)\n")
    else:
        output.append("RNN model (replace heads -- left)\n")
    output.append("-" * 60 + "\n")
    mean_rank = np.mean(ranklist)
    output.append("mean rank = %6.4f\n" % mean_rank)
    fil_mean_rank = np.mean(ranklist_filter)
    output.append("fil mean rank = %6.4f\n" % fil_mean_rank)
    p_10 = (p_10 * 100) / ndata
    p_1 = (p_1 * 100) / ndata
    p_3 = (p_3 * 100) / ndata
    p_5 = (p_5 * 100) / ndata
    p_7 = (p_7 * 100) / ndata
    output.append("unfiltered:\tp_1 = %6.4f\t\tp_3 = %6.4f\t\tp_5 = %6.4f\t\tp_7 = %6.4f\t\tP@10 = %6.4f\n" % (
    p_1, p_3, p_5, p_7, p_10))

    p_10_filter = (p_10_filter * 100) / ndata
    p_1_filter = (p_1_filter * 100) / ndata
    p_3_filter = (p_3_filter * 100) / ndata
    p_5_filter = (p_5_filter * 100) / ndata
    p_7_filter = (p_7_filter * 100) / ndata
    output.append("filtered:\t\tp_1 = %6.4f\t\tp_3 = %6.4f\t\tp_5 = %6.4f\t\tp_7 = %6.4f\t\tP@10 = %6.4f\n" %
                  (p_1_filter, p_3_filter, p_5_filter, p_7_filter, p_10_filter))

    ISOTIMEFORMAT = '%m%d%H%M'
    timestr = datetime.datetime.now()
    filename = "./results/" + str(mode) + '_' + str(timestr.strftime(ISOTIMEFORMAT)) + ".txt"
    f = open(filename, 'w')
    f.writelines(output)
    f.close()


def calc_fscore(index = None):
    # 1. read test data
    trainfile = "../../../../data/kbase/processed/blogcatalog/percent/train/label_train_"+str(index)
    testfile = "../../../../data/kbase/processed/blogcatalog/percent/test/label_test_"+str(index)

    ftest = open(testfile, 'r')
    nent =10312   # 3890 10312
    y_test = [[] for x in range(nent)]
    for line in ftest:
        [h, r, t] = line.strip().split('\t')
        h = int(h)
        t = int(t)
        if t not in y_test[h]:
            y_test[h].append(t)
    ftest.close()

    ftrain = open(trainfile, 'r')
    for line in ftrain:
        [h, r, t] = line.strip().split('\t')
        h = int(h)
        t = int(t)
        if t not in y_test[h]:
            y_test[h].append(t)
    ftrain.close()

    # 2. read pred score
    pklfile = '/home/lanco/Work/Source/kbase/source/kbase/public/GEN/results/percent_batch/gen_blog_'+str(index)+'.pkl'
    fin = open(pklfile, 'rb')
    results = pickle.load(fin)
    fin.close()
    triples = results['triples']
    softmax = results['softmax']

    preds = [[] for x in range(nent)]

    for i in range(len(triples)):
        h = int(triples[i][0])
        r = int(triples[i][1])
        t = int(triples[i][2])

        # if r != 10351:
        #     continue

        predlist = softmax[i][0]
        top_k = len(y_test[h])
        top_k_list = heapq.nlargest(top_k, range(len(predlist)), predlist.__getitem__)
        top_k_list = [j for j in top_k_list]
        preds[h] = top_k_list

    #
    # 3. calculate f1_score
    averages = ["micro", "macro"]
    results = {}

    indexs = []
    for i in range(len(preds)):
        if len(preds[i]) != 0:
            indexs.append(i)

    y_test = np.take(np.array(y_test), indexs).tolist()
    preds = np.take(np.array(preds), indexs).tolist()

    # print(y_test)
    # print(preds)
    for average in averages:
        results[average] = f1_score(y_test, preds, average=average)

    print('-------------------')
    print('precent: '+str(index))
    print(results)
    print('-------------------')


