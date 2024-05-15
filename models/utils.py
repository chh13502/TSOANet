import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import math
import torch.nn.init as init
import random


class CrossEntropy(nn.Module):
    def __init__(self, n_labels):
        super(CrossEntropy, self).__init__()
        self.num_class = n_labels

    def forward(self, inputs, targets):
        logEps = 1e-8
        cross_entropy = -(targets * torch.log(inputs + logEps) + (1. - targets) * torch.log(1. - inputs + logEps))

        prediction_loss = torch.sum(cross_entropy, dim=1)
        loss = torch.mean(prediction_loss)
        return loss

def pad_time(seq_time_step, options):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    return seq_time_step


def get_maxCode(seqs):
    length_code = []
    for seq in seqs:
        for code_set in seq:
            length_code.append(len(code_set))
    length_code = np.array(length_code)
    maxcode = np.max(length_code)
    return maxcode


def pad_matrix_new(seq_med_codes, seq_labtest_codes, seq_diag_codes, seq_proc_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diag_codes])
    len_set = set(lengths.tolist())

    n_samples = len(seq_diag_codes)
    n_med_codes = options['n_med_codes']
    n_labtest_codes = options['n_labtest_codes']
    n_diag_codes = options['n_diag_codes']
    n_proc_codes = options['n_proc_codes']
    maxlen = np.max(lengths)
    lengths_code = []
    maxcode_med = get_maxCode(seq_med_codes)
    maxcode_diag = get_maxCode(seq_diag_codes)


    batch_med_codes = np.zeros((n_samples, maxlen, maxcode_med), dtype=np.int64) + n_med_codes
    batch_med_mask_code = np.zeros((n_samples, maxlen, maxcode_med), dtype=np.float32)

    batch_diag_codes = np.zeros((n_samples, maxlen, maxcode_diag), dtype=np.int64) + n_diag_codes
    batch_diag_mask_code = np.zeros((n_samples, maxlen, maxcode_diag), dtype=np.float32)

    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    # Medic
    for bid, seq in enumerate(seq_med_codes):               
        for pid, subseq in enumerate(seq):                  
            for tid, code in enumerate(subseq):             
                batch_med_codes[bid, pid, tid] = code
                batch_med_mask_code[bid, pid, tid] = 1

    # Labtest
    if options['dataset'] == 'mimic_data' or options['dataset'] == 'eicu_data':
        maxcode_labtest = get_maxCode(seq_labtest_codes)
        batch_labtest_codes = np.zeros((n_samples, maxlen, maxcode_labtest), dtype=np.int64) + n_labtest_codes
        batch_labtest_mask_code = np.zeros((n_samples, maxlen, maxcode_labtest), dtype=np.float32)
        for bid, seq in enumerate(seq_labtest_codes):
            for pid, subseq in enumerate(seq):
                for tid, code in enumerate(subseq):
                    batch_labtest_codes[bid, pid, tid] = code
                    batch_labtest_mask_code[bid, pid, tid] = 1
    else:
        batch_labtest_codes = []
        batch_labtest_mask_code = []

    # Diag
    for bid, seq in enumerate(seq_diag_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diag_codes[bid, pid, tid] = code
                batch_diag_mask_code[bid, pid, tid] = 1

    # Proc
    if options['dataset'] == 'mimic_data' or options['dataset'] == 'DAPS_data':
        maxcode_proc = get_maxCode(seq_proc_codes)
        batch_proc_codes = np.zeros((n_samples, maxlen, maxcode_proc), dtype=np.int64) + n_proc_codes
        batch_proc_mask_code = np.zeros((n_samples, maxlen, maxcode_proc), dtype=np.float32)
        for bid, seq in enumerate(seq_proc_codes):
            for pid, subseq in enumerate(seq):
                for tid, code in enumerate(subseq):
                    batch_proc_codes[bid, pid, tid] = code
                    batch_proc_mask_code[bid, pid, tid] = 1
    else:
        batch_proc_codes = []
        batch_proc_mask_code = []


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1       
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1     

    # one-hot
    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_labels, batch_mask, \
           batch_mask_final, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code


def set_keep_order(sequences):
    set_seq = list(set(sequences))
    set_seq.sort(key=sequences.index)
    return set_seq


def padInputWithTime_new(seqs, labels, times, g_times, options):
    lengths = np.array([len(seq) for seq in seqs])
    n_sample = len(seqs)
    maxlen = np.max(lengths)
    numClass = options['n_labels']
    med_x = []
    labtest_x = []
    diag_x = []
    proc_x = []
    y = np.zeros((n_sample, numClass))
    original_y = []
    max_len = []
    if options['dataset'] == 'mimic_data':
        # remove tuple
        for idx, seq in enumerate(seqs):        
            tmp_med_x = []
            tmp_labtest_x = []
            tmp_diag_x = []
            tmp_proc_x = []
            for subseq in seq:                  
                diag_code = subseq[0]
                proc_code = subseq[1]
                med_code = subseq[2]
                labtest_code = subseq[3]
                tmp_med_x.append(med_code)
                tmp_labtest_x.append(labtest_code)
                tmp_diag_x.append(diag_code)
                tmp_proc_x.append(proc_code)
            med_x.append(tmp_med_x)
            labtest_x.append(tmp_labtest_x)
            diag_x.append(tmp_diag_x)
            proc_x.append(tmp_proc_x)
            ## *_x = [patient_1, patient_2, ...,]
            ## *patient = [*_code1, *_2, ...]

            # y: one-hot
            # original_y: code
        for yvec, label in zip(y, labels):
            last_label  = label
            med_code = last_label[2]
            diag_code = last_label[0]
            proc_code = last_label[1]
            labtest_code = last_label[3]
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predProc']:
                yvec[proc_code] = 1.
                original_y.append(proc_code)
            elif options['predLabtest']:
                yvec[labtest_code] = 1.
                original_y.append(labtest_code)
            else:
                yvec[med_code] = 1.
                original_y.append(med_code)
        
    elif options['dataset'] == 'eicu_data':
        for idx, seq in enumerate(seqs):
            tmp_med_x = []
            tmp_labtest_x = []
            tmp_diag_x = []
            for index, subseq in enumerate(seq):
                med_code = set_keep_order(subseq[0])
                labtest_code = set_keep_order(subseq[1])
                diag_code = set_keep_order(subseq[2])
                tmp_med_x.append(med_code)
                tmp_labtest_x.append(labtest_code)
                tmp_diag_x.append(diag_code)
            med_x.append(tmp_med_x)
            labtest_x.append(tmp_labtest_x)
            diag_x.append(tmp_diag_x)
        for yvec , label in zip(y, labels):
            last_label = label
            med_code = set_keep_order(last_label[0])
            labtest_code = set_keep_order(last_label[1])
            diag_code = set_keep_order(last_label[2])
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predLabtest']:
                yvec[labtest_code] = 1
                original_y.append(labtest_code)
            else:
                yvec[med_code] = 1
                original_y.append(med_code)
    elif options['dataset'] == 'DAPS_data':
        for idx, seq in enumerate(seqs):
            tmp_med_x = []
            tmp_proc_x = []
            tmp_diag_x = []
            for index, subseq in enumerate(seq):
                med_code = set_keep_order(subseq[2])
                proc_code = set_keep_order(subseq[1])
                diag_code = set_keep_order(subseq[0])
                tmp_med_x.append(med_code)
                tmp_proc_x.append(proc_code)
                tmp_diag_x.append(diag_code)
            med_x.append(tmp_med_x)
            proc_x.append(tmp_proc_x)
            diag_x.append(tmp_diag_x)
        for yvec , label in zip(y, labels):
            last_label = label
            med_code = set_keep_order(last_label[2])
            proc_code = set_keep_order(last_label[1])
            diag_code = set_keep_order(last_label[0])
            if options['predDiag']:
                yvec[diag_code] = 1
                original_y.append(diag_code)
            elif options['predProc']:
                yvec[proc_code] = 1
                original_y.append(proc_code)
            else:
                yvec[med_code] = 1
                original_y.append(med_code)
    
    lengths = np.array(lengths)
    return med_x, labtest_x, diag_x, proc_x, times, g_times, y, original_y



##
class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

##
class ScaledDotProductAttention_dim4(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention_dim4, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=3)            

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # q: B, T, X_LEN, H
        x = q
        attention = torch.matmul(q, k.transpose(2, 3))  # attention: B, 1, X_len, H
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context + x
        return context, attention

class ScaledDotProductAttention_TimeGate(nn.Module):
    """Scaled dot-product attention mechanism."""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention_TimeGate, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=3)            

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # q: B, T, X_LEN, H
        x = q
        attention = torch.matmul(q, k.transpose(2, 3))  # attention: B, 1, X_len, H
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        return context, attention


##
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)            

    def forward(self, q, k, v, scale=None, attn_mask=None):
        x = q
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        context = context + x
        return context, attention

##
class ScaledDotProductAttention_dim1(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention_dim1, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)           

    def forward(self, q, k, v, scale=None, attn_mask=None):
        x = q
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        context = context + x
        return context, attention


## 位置编码
class PositionalEncoding(nn.Module):
    """
    make up a lookup table
    transform position_index to positional_embedding
    """
    def __init__(self, d_model, max_seq_len, options):
        """
            model_d: embedding dimensions
        """

        super(PositionalEncoding, self).__init__()
        self.options= options

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):


        max_len = torch.max(input_len)

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = torch.from_numpy(pos).long().to(self.options['device'])

        return self.position_encoding(input_pos), input_pos

class TimeEncoder(nn.Module):
    def __init__(self, batch_size, hidden_size,):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, seq_time_step, final_queries, options, mask):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        return torch.softmax(selection_feature, 1)
