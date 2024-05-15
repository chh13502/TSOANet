import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
import random
from models.utils import *

def recallTop(y_true, y_pred, rank=[10, 20, 30]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / len(set(codes)))
        recall.append(thisOne)
    return (np.array(recall)).mean(axis=0).tolist()


def precisionTop(y_true, y_pred, rank=[10, 20, 30]):
    precision = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / len(set(tops)))
        precision.append(thisOne)
    return (np.array(precision)).mean(axis=0).tolist()


def F1Top(y_true, y_pred, rank=[10, 20, 30]):
    recall_top = recallTop(y_true, y_pred, rank)
    precision_top = precisionTop(y_true, y_pred, rank)
    F1 = list()
    for i in range(len(rank)):
        tmp_f1 = (2 * precision_top[i] * recall_top[i]) / (precision_top[i] + recall_top[i])
        F1.append(tmp_f1)
    return F1


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def load_data(seqFile, labelFile, timeFile):
    train_set_x = pickle.load(open(seqFile + '.train', 'rb'))
    valid_set_x = pickle.load(open(seqFile + '.valid', 'rb'))
    test_set_x = pickle.load(open(seqFile + '.test', 'rb'))
    train_set_s = pickle.load(open(seqFile + '.static_train', 'rb'))
    valid_set_s = pickle.load(open(seqFile + '.static_valid', 'rb'))
    test_set_s = pickle.load(open(seqFile + '.static_test', 'rb'))
    train_set_y = pickle.load(open(labelFile + '.train', 'rb'))
    valid_set_y = pickle.load(open(labelFile + '.valid', 'rb'))
    test_set_y = pickle.load(open(labelFile + '.test', 'rb'))
    train_set_t = pickle.load(open(timeFile + '.train_new', 'rb'))
    valid_set_t = pickle.load(open(timeFile + '.valid_new', 'rb'))
    test_set_t = pickle.load(open(timeFile + '.test_new', 'rb'))
    train_set_gt = pickle.load(open(timeFile + '.train_global_times', 'rb'))
    valid_set_gt = pickle.load(open(timeFile + '.valid_global_times', 'rb'))
    test_set_gt = pickle.load(open(timeFile + '.test_global_times', 'rb'))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    train_set_s = [train_set_s[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    valid_set_s = [valid_set_s[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    test_set_s = [test_set_s[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]
        train_set_gt = [train_set_gt[i] for i in train_sorted_index]
        valid_set_gt = [valid_set_gt[i] for i in valid_sorted_index]
        test_set_gt = [test_set_gt[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t, train_set_gt, train_set_s)
    valid_set = (valid_set_x, valid_set_y, valid_set_t, valid_set_gt, valid_set_s)
    test_set = (test_set_x, test_set_y, test_set_t, test_set_gt, test_set_s)

    return train_set, valid_set, test_set

# 剪裁长度 设立标志位<CLS>
def adjust_input_new(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step,
                     batch_global_time, max_len, n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, options):
    batch_time_step = copy.deepcopy(batch_time_step)
    batch_global_time = copy.deepcopy(batch_global_time)
    batch_med_codes = copy.deepcopy(batch_med_codes)
    batch_labtest_codes = copy.deepcopy(batch_labtest_codes)
    batch_diag_codes = copy.deepcopy(batch_diag_codes)
    batch_proc_codes = copy.deepcopy(batch_proc_codes)
    for ind in range(len(batch_diag_codes)):  # 对于每个病人
        if len(batch_diag_codes[ind]) > max_len:
            batch_diag_codes[ind] = batch_diag_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
            batch_global_time[ind] = batch_global_time[ind][-(max_len):]
            batch_med_codes[ind] = batch_med_codes[ind][-(max_len):]
        if options['dataset'] == 'eicu_data':
            if len(batch_labtest_codes[ind]) > max_len:
                batch_labtest_codes[ind] = batch_labtest_codes[ind][-(max_len):]
            batch_labtest_codes[ind].append([n_labtest_codes - 1])

        if options['dataset'] == 'DAPS_data':
            if len(batch_proc_codes[ind]) > max_len:
                batch_proc_codes[ind] = batch_proc_codes[ind][-(max_len):]
            batch_proc_codes[ind].append([n_proc_codes - 1])

        if options['dataset'] == 'mimic_data':
            if len(batch_labtest_codes[ind]) > max_len:
                batch_labtest_codes[ind] = batch_labtest_codes[ind][-(max_len):]
            batch_labtest_codes[ind].append([n_labtest_codes - 1])

            if len(batch_proc_codes[ind]) > max_len:
                batch_proc_codes[ind] = batch_proc_codes[ind][-(max_len):]
            batch_proc_codes[ind].append([n_proc_codes - 1])

        batch_time_step[ind].append(0)
        batch_global_time[ind].append(0)
        batch_med_codes[ind].append([n_med_codes - 1])
        batch_diag_codes[ind].append([n_diag_codes - 1])
        # batch_labtest_codes[ind].append([n_labtest_codes - 1])

    return batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time



def calculate_cost_tran(model, data, options, max_len, loss_function=F.cross_entropy):
    model.eval()
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    total_pred = []
    total_true = []
    for index in range(n_batches):
        batchX = data[0][batch_size * index: batch_size * (index + 1)]
        batchY = data[1][batch_size * index: batch_size * (index + 1)]
        batchS = data[-1][batch_size * index: batch_size * (index + 1)]
        batchT = data[2][batch_size * index: batch_size * (index + 1)]
        batchGT = data[3][batch_size * index: batch_size * (index + 1)]

        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, \
        batch_labels, batch_original_y = padInputWithTime_new(batchX, batchY, batchT, batchGT, options)

        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, \
        batch_time_step, batch_global_time = adjust_input_new(
            batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time,
            max_len, options['n_med_codes'], options['n_labtest_codes'],
            options['n_diag_codes'], options['n_proc_codes'], options)
        lengths = np.array([len(seq) for seq in batch_diag_codes])
        maxlen = np.max(lengths)
        logit, labels, self_attention = model(batch_med_codes, batch_labtest_codes, batch_diag_codes,
                                                batch_proc_codes, batch_time_step, batch_global_time, batch_labels,
                                                options, maxlen, batchS)

        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()

        pred_score = logit.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        total_pred.append(pred_score)
        total_true.append(y_true)
    total_pred_value = np.concatenate(total_pred)
    total_true_value = np.concatenate(total_true)
    total_avg_auc_micro = roc_auc_score(total_true_value, total_pred_value, average='micro')
    total_avg_aupr_micro = average_precision_score(total_true_value, total_pred_value, average='micro')

    model.train()
    return cost_sum / n_batches, total_avg_auc_micro, total_avg_aupr_micro


