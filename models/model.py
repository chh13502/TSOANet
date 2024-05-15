import pickle

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from models import utils
from models.utils import *
import copy


#MIMIC
class Encoder_MIMIC(nn.Module): 
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                vocab_size_proc,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(Encoder_MIMIC, self).__init__()


        self.model_dim = model_dim
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding_med = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_med = 1 / math.sqrt(vocab_size_med)
        init.uniform_(self.bias_embedding_med, -bound_med, bound_med)
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        self.bias_embedding_labtest = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_labtest = 1 / math.sqrt(vocab_size_labtest)
        init.uniform_(self.bias_embedding_labtest, -bound_labtest, bound_labtest)
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        self.bias_embedding_diag = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_diag = 1 / math.sqrt(vocab_size_diag)
        init.uniform_(self.bias_embedding_diag, -bound_diag, bound_diag)
        self.pre_embedding_proc = Embedding(vocab_size_proc, model_dim)
        self.bias_embedding_proc = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_proc = 1 / math.sqrt(vocab_size_proc)
        init.uniform_(self.bias_embedding_proc, -bound_proc, bound_proc)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.att_func_med = ScaledDotProductAttention()
        self.att_func_labtest = ScaledDotProductAttention()
        self.att_func_diag = ScaledDotProductAttention()
        self.att_func_proc = ScaledDotProductAttention()
        self.att_func = ScaledDotProductAttention()

        self.embed_self_att = ScaledDotProductAttention_dim4()

        self.tanh = nn.Tanh()
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.dropout_emb = nn.Dropout(dp)

        self.time_gate_attn = ScaledDotProductAttention_TimeGate()
        self.time_gate_med_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_labtest_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_diag_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_proc_layer = torch.nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, med_codes, labtest_codes, diag_codes, proc_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, proc_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2)
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)   # B,T,1,H
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,M_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        output_proc = (self.pre_embedding_proc(proc_codes) * proc_mask_code)

        scale = 1/math.sqrt(options['hidden_size']+1e-7)

        # time_feature: B, T, 1, H
        # output_X: B, T, X_len, H
        # gate_X: B, T, 1, H
        gate_med, _ = self.time_gate_attn(time_feature, output_med, output_med, scale=scale)
        gate_med = self.sigmoid(self.time_gate_med_layer(gate_med))
        output_med = gate_med*output_med
        gate_labtest, _ = self.time_gate_attn(time_feature, output_labtest, output_labtest, scale=scale)
        gate_labtest = self.sigmoid(self.time_gate_labtest_layer(gate_labtest))
        output_labtest = gate_labtest*output_labtest
        gate_diag, _ = self.time_gate_attn(time_feature, output_diag, output_diag, scale=scale)
        gate_diag = self.sigmoid(self.time_gate_diag_layer(gate_diag))
        output_diag = gate_diag*output_diag
        gate_proc, _ = self.time_gate_attn(time_feature, output_proc, output_proc, scale=scale)
        gate_proc = self.sigmoid(self.time_gate_proc_layer(gate_proc))
        output_proc = gate_proc*output_proc

        output_med = output_med.view(batch_size, -1, self.model_dim)  # B,T*M_len,H
        output_labtest = output_labtest.view(batch_size, -1, self.model_dim)
        output_diag = output_diag.view(batch_size, -1, self.model_dim)
        output_proc = output_proc.view(batch_size, -1, self.model_dim)

        output_med, _ = self.att_func(output_med, output_med, output_med, scale=scale)
        output_labtest, _ = self.att_func(output_labtest, output_labtest, output_labtest, scale=scale)
        output_diag, _ = self.att_func(output_diag, output_diag, output_diag, scale=scale)
        output_proc, _ = self.att_func(output_proc, output_proc, output_proc, scale=scale)

        output_med = output_med.view(batch_size, time_step, -1, self.model_dim)
        output_labtest = output_labtest.view(batch_size, time_step, -1, self.model_dim)
        output_diag = output_diag.view(batch_size, time_step, -1, self.model_dim)
        output_proc = output_proc.view(batch_size, time_step, -1, self.model_dim)

        merge_output_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature),dim=2).view(batch_size, time_step, -1,self.model_dim)  # B, T, M_len+L_len+P_len+D_len, H
        merge_output, _ = self.embed_self_att(merge_output_aux, merge_output_aux, merge_output_aux, scale=scale)
        split_size = [output_med.shape[-2], output_labtest.shape[-2], output_diag.shape[-2], output_proc.shape[-2], time_feature.shape[-2]]
        output_med, output_labtest, output_diag, output_proc, time_feature = merge_output.split(split_size, dim=2)


        if options['predDiag']:
            merge_output_diag_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)   # B*T, M_len+L_len+P_len+D_len, H
            output_diag_trans = output_diag.view(batch_size*time_step, -1, self.model_dim)  # B*T, D_len, H
            diag_att, _ = self.att_func_diag(output_diag_trans, merge_output_diag_aux, merge_output_diag_aux)
            output_embed_diag = diag_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_diag  # B, T, H
            output_embed_diag = output_embed_diag + output_pos
            output_embed = output_embed_diag

        elif options['predLabtest']:
            merge_output_labtest_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_labtest_trans = output_labtest.view(batch_size*time_step, -1, self.model_dim)
            labtest_att, _ = self.att_func_labtest(output_labtest_trans, merge_output_labtest_aux, merge_output_labtest_aux)
            output_embed_labtest = labtest_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_labtest
            output_embed_labtest = output_embed_labtest + output_pos
            output_embed = output_embed_labtest

        elif options['predProc']:
            merge_output_proc_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_proc_trans = output_proc.view(batch_size*time_step, -1, self.model_dim)
            proc_att, _ = self.att_func_proc(output_proc_trans, merge_output_proc_aux, merge_output_proc_aux)
            output_embed_proc = proc_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_proc
            output_embed_proc = output_embed_proc + output_pos
            output_embed = output_embed_proc

        else:
            merge_output_med_aux = torch.cat((output_med, output_labtest, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_med_trans = output_med.view(batch_size*time_step, -1, self.model_dim)
            med_att, _ = self.att_func_med(output_med_trans, merge_output_med_aux, merge_output_med_aux)
            output_embed_med = med_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_med
            output_embed_med = output_embed_med + output_pos
            output_embed = output_embed_med

        output = self.dropout_emb(output_embed)

        return output

#eICU
class Encoder_eicu(nn.Module):  
    def __init__(self,
                vocab_size_med,
                vocab_size_labtest,
                vocab_size_diag,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(Encoder_eicu, self).__init__()

        self.model_dim = model_dim
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding_med = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_med = 1 / math.sqrt(vocab_size_med)
        init.uniform_(self.bias_embedding_med, -bound_med, bound_med)
        self.pre_embedding_labtest = Embedding(vocab_size_labtest, model_dim)
        self.bias_embedding_labtest = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_labtest = 1 / math.sqrt(vocab_size_labtest)
        init.uniform_(self.bias_embedding_labtest, -bound_labtest, bound_labtest)
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        self.bias_embedding_diag = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_diag = 1 / math.sqrt(vocab_size_diag)
        init.uniform_(self.bias_embedding_diag, -bound_diag, bound_diag)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.att_func_med = ScaledDotProductAttention()
        self.att_func_labtest = ScaledDotProductAttention()
        self.att_func_diag = ScaledDotProductAttention()
        self.att_func_proc = ScaledDotProductAttention()
        self.att_func = ScaledDotProductAttention()

        self.embed_self_att = ScaledDotProductAttention_dim4()

        self.tanh = nn.Tanh()
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.dropout_emb = nn.Dropout(dp)

        self.time_gate_attn = ScaledDotProductAttention_TimeGate()
        self.time_gate_med_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_labtest_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_diag_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_proc_layer = torch.nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, med_codes, labtest_codes, diag_codes, mask, med_mask_code, labtest_mask_code, diag_mask_code, seq_time_step, input_len, options):
        seq_time_step = seq_time_step.unsqueeze(2)
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)   # B,T,1,H
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,M_len,H
        output_labtest = (self.pre_embedding_labtest(labtest_codes) * labtest_mask_code)
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)

        scale = 1/math.sqrt(options['hidden_size']+1e-7)

        # time_feature: B, T, 1, H
        # output_X: B, T, X_len, H
        # gate_X: B, T, 1, H
        gate_med, _ = self.time_gate_attn(time_feature, output_med, output_med, scale=scale)
        gate_med = self.sigmoid(self.time_gate_med_layer(gate_med))
        output_med = gate_med*output_med
        gate_labtest, _ = self.time_gate_attn(time_feature, output_labtest, output_labtest, scale=scale)
        gate_labtest = self.sigmoid(self.time_gate_labtest_layer(gate_labtest))
        output_labtest = gate_labtest*output_labtest
        gate_diag, _ = self.time_gate_attn(time_feature, output_diag, output_diag, scale=scale)
        gate_diag = self.sigmoid(self.time_gate_diag_layer(gate_diag))
        output_diag = gate_diag*output_diag

        output_med = output_med.view(batch_size, -1, self.model_dim)  # B,T*M_len,H
        output_labtest = output_labtest.view(batch_size, -1, self.model_dim)
        output_diag = output_diag.view(batch_size, -1, self.model_dim)

        output_med, _ = self.att_func(output_med, output_med, output_med, scale=scale)
        output_labtest, _ = self.att_func(output_labtest, output_labtest, output_labtest, scale=scale)
        output_diag, _ = self.att_func(output_diag, output_diag, output_diag, scale=scale)

        output_med = output_med.view(batch_size, time_step, -1, self.model_dim)
        output_labtest = output_labtest.view(batch_size, time_step, -1, self.model_dim)
        output_diag = output_diag.view(batch_size, time_step, -1, self.model_dim)


        merge_output_aux = torch.cat((output_med, output_labtest, output_diag, time_feature),dim=2).view(batch_size, time_step, -1,self.model_dim)  # B, T, M_len+L_len+P_len+D_len, H
        merge_output, _ = self.embed_self_att(merge_output_aux, merge_output_aux, merge_output_aux, scale=scale)
        split_size = [output_med.shape[-2], output_labtest.shape[-2], output_diag.shape[-2], time_feature.shape[-2]]
        output_med, output_labtest, output_diag, time_feature = merge_output.split(split_size, dim=2)

        if options['predDiag']:
            merge_output_diag_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)   # B*T, M_len+L_len+P_len+D_len, H
            output_diag_trans = output_diag.view(batch_size*time_step, -1, self.model_dim)  # B*T, D_len, H
            diag_att, _ = self.att_func_diag(output_diag_trans, merge_output_diag_aux, merge_output_diag_aux)
            output_embed_diag = diag_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_diag  # B, T, H
            output_embed_diag = output_embed_diag + output_pos
            output_embed = output_embed_diag

        elif options['predLabtest']:
            merge_output_labtest_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_labtest_trans = output_labtest.view(batch_size*time_step, -1, self.model_dim)
            labtest_att, _ = self.att_func_labtest(output_labtest_trans, merge_output_labtest_aux, merge_output_labtest_aux)
            output_embed_labtest = labtest_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_labtest
            output_embed_labtest = output_embed_labtest + output_pos
            output_embed = output_embed_labtest

        else:
            merge_output_med_aux = torch.cat((output_med, output_labtest, output_diag, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_med_trans = output_med.view(batch_size*time_step, -1, self.model_dim)
            med_att, _ = self.att_func_med(output_med_trans, merge_output_med_aux, merge_output_med_aux)
            output_embed_med = med_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_med
            output_embed_med = output_embed_med + output_pos
            output_embed = output_embed_med

        output = self.dropout_emb(output_embed)
        return output

#DAPS
class Encoder_DAPS(nn.Module):  
    def __init__(self,
                vocab_size_med,
                vocab_size_diag,
                vocab_size_proc,
                max_seq_len,
                dp,
                device,
                model_dim=256,
                num_layers=1,
                num_heads=4,
                ffn_dim=1024,
                dropout=0.0,
                options=None):
        super(Encoder_DAPS, self).__init__()

        self.model_dim = model_dim
        self.pre_embedding_med = Embedding(vocab_size_med, model_dim)
        self.bias_embedding_med = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_med = 1 / math.sqrt(vocab_size_med)
        init.uniform_(self.bias_embedding_med, -bound_med, bound_med)
        self.pre_embedding_diag = Embedding(vocab_size_diag, model_dim)
        self.bias_embedding_diag = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_diag = 1 / math.sqrt(vocab_size_diag)
        init.uniform_(self.bias_embedding_diag, -bound_diag, bound_diag)
        self.pre_embedding_proc = Embedding(vocab_size_proc, model_dim)
        self.bias_embedding_proc = torch.nn.Parameter(torch.Tensor(model_dim))
        bound_proc = 1 / math.sqrt(vocab_size_proc)
        init.uniform_(self.bias_embedding_proc, -bound_proc, bound_proc)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, options)
        self.att_func_med = ScaledDotProductAttention()
        self.att_func_labtest = ScaledDotProductAttention()
        self.att_func_diag = ScaledDotProductAttention()
        self.att_func_proc = ScaledDotProductAttention()
        self.att_func = ScaledDotProductAttention()

        self.embed_self_att = ScaledDotProductAttention_dim4()

        self.tanh = nn.Tanh()
        self.time_layer = torch.nn.Linear(64, model_dim)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.dropout_emb = nn.Dropout(dp)

        self.time_gate_attn = ScaledDotProductAttention_TimeGate()
        self.time_gate_med_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_labtest_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_diag_layer = torch.nn.Linear(model_dim, model_dim)
        self.time_gate_proc_layer = torch.nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, med_codes, diag_codes, proc_codes, mask, med_mask_code, diag_mask_code, proc_mask_code, seq_time_step, input_len, options):
        """ time encoding
        time_feature = W_2(1-tanh((W_1*t+b_1)^2))+b_2
        """
        seq_time_step = seq_time_step.unsqueeze(2)
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature).unsqueeze(2)   # B,T,1,H
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        batch_size, time_step = med_codes.shape[0], med_codes.shape[1]
        output_med = (self.pre_embedding_med(med_codes) * med_mask_code) # B,T,M_len,H
        output_diag = (self.pre_embedding_diag(diag_codes) * diag_mask_code)
        output_proc = (self.pre_embedding_proc(proc_codes) * proc_mask_code)

        scale = 1/math.sqrt(options['hidden_size']+1e-7)

        # time_feature: B, T, 1, H
        # output_X: B, T, X_len, H
        # gate_X: B, T, 1, H
        gate_med, _ = self.time_gate_attn(time_feature, output_med, output_med, scale=scale)
        gate_med = self.sigmoid(self.time_gate_med_layer(gate_med))
        output_med = gate_med*output_med
        gate_diag, _ = self.time_gate_attn(time_feature, output_diag, output_diag, scale=scale)
        gate_diag = self.sigmoid(self.time_gate_diag_layer(gate_diag))
        output_diag = gate_diag*output_diag
        gate_proc, _ = self.time_gate_attn(time_feature, output_proc, output_proc, scale=scale)
        gate_proc = self.sigmoid(self.time_gate_proc_layer(gate_proc))
        output_proc = gate_proc*output_proc



        output_med = output_med.view(batch_size, -1, self.model_dim)  # B,T*M_len,H
        output_diag = output_diag.view(batch_size, -1, self.model_dim)
        output_proc = output_proc.view(batch_size, -1, self.model_dim)

        output_med, _ = self.att_func(output_med, output_med, output_med, scale=scale)
        output_diag, _ = self.att_func(output_diag, output_diag, output_diag, scale=scale)
        output_proc, _ = self.att_func(output_proc, output_proc, output_proc, scale=scale)

        output_med = output_med.view(batch_size, time_step, -1, self.model_dim)
        output_diag = output_diag.view(batch_size, time_step, -1, self.model_dim)
        output_proc = output_proc.view(batch_size, time_step, -1, self.model_dim)

        merge_output_aux = torch.cat((output_med, output_diag, output_proc, time_feature),dim=2).view(batch_size, time_step, -1,self.model_dim)  # B, T, M_len+L_len+P_len+D_len, H
        merge_output, _ = self.embed_self_att(merge_output_aux, merge_output_aux, merge_output_aux, scale=scale)
        split_size = [output_med.shape[-2],  output_diag.shape[-2], output_proc.shape[-2], time_feature.shape[-2]]
        output_med, output_diag, output_proc, time_feature = merge_output.split(split_size, dim=2)

        if options['predDiag']:
            merge_output_diag_aux = torch.cat((output_med, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)   # B*T, M_len+L_len+P_len+D_len, H
            output_diag_trans = output_diag.view(batch_size*time_step, -1, self.model_dim)  # B*T, D_len, H
            diag_att, _ = self.att_func_diag(output_diag_trans, merge_output_diag_aux, merge_output_diag_aux)
            output_embed_diag = diag_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_diag  # B, T, H
            output_embed_diag = output_embed_diag + output_pos
            output_embed = output_embed_diag


        elif options['predProc']:
            merge_output_proc_aux = torch.cat((output_med, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_proc_trans = output_proc.view(batch_size*time_step, -1, self.model_dim)
            proc_att, _ = self.att_func_proc(output_proc_trans, merge_output_proc_aux, merge_output_proc_aux)
            output_embed_proc = proc_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_proc
            output_embed_proc = output_embed_proc + output_pos
            output_embed = output_embed_proc

        else:
            merge_output_med_aux = torch.cat((output_med, output_diag, output_proc, time_feature), dim=2).view(batch_size*time_step, -1, self.model_dim)
            output_med_trans = output_med.view(batch_size*time_step, -1, self.model_dim)
            med_att, _ = self.att_func_med(output_med_trans, merge_output_med_aux, merge_output_med_aux)
            output_embed_med = med_att.sum(dim=1).view(batch_size, time_step, -1) + self.bias_embedding_med
            output_embed_med = output_embed_med + output_pos
            output_embed = output_embed_med

        output = self.dropout_emb(output_embed)

        return output


class TSOANet(nn.Module):
    def __init__(self, n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, batch_size, options):
        super(TSOANet, self).__init__()
        self.time_encoder = TimeEncoder(batch_size, options['hidden_size'])
        if options['dataset'] == 'mimic_data':
            self.feature_encoder = Encoder_MIMIC(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, n_proc_codes+1, 31, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        elif options['dataset'] == 'eicu_data':
            self.feature_encoder = Encoder_eicu(n_med_codes+1, n_labtest_codes+1, n_diag_codes+1, 11, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        elif options['dataset'] == 'DAPS_data':
            self.feature_encoder = Encoder_DAPS(n_med_codes+1, n_diag_codes+1, n_proc_codes+1, 29, options['dp'], options['device'], model_dim=options['visit_size'], options=options)
        self.self_layer = torch.nn.Linear(options['hidden_size'], 1)
        self.classify_layer = torch.nn.Linear(options['hidden_size'], options['n_labels'])
        self.quiry_layer = torch.nn.Linear(options['hidden_size'], options['hidden_size'])
        self.quiry_weight_layer = torch.nn.Linear(options['hidden_size'], 2)
        self.quiry_2weight_layer = torch.nn.Linear(options['hidden_size'],2)
        self.relu = nn.ReLU(inplace=True)
        dropout_rate = options['dp']
        self.dropout = nn.Dropout(dropout_rate)
        self.static_trans = nn.Linear(options['static_dim'], options['hidden_size'])
        self.selfatt = ScaledDotProductAttention(0.1)

    def get_self_attention(self, features, mask):
        att_output, att_score = self.selfatt(features, features, features)
        attention = torch.softmax(self.self_layer(att_output).masked_fill(mask, -np.inf), dim=1)    # 实际上是用0来填
        return attention

    def forward(self, batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, seq_time_step, batch_global_time, batch_labels, options, maxlen, batchS):
        # batch_XX_codes: [batch_size, visit_length, code_num(one-hot)]
        # seq_time_step: [batch_size, visit_length] the number of days to the final visit
        # batch_labels: [batch_size] 0 negative 1 positive
        seq_time_step = np.array(list(utils.pad_time(seq_time_step, options)))
        batch_global_time = np.array(list(utils.pad_time(batch_global_time, options)))
        # align code
        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_labels, batch_mask, batch_mask_final, batch_med_mask_code, \
            batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code = utils.pad_matrix_new(batch_med_codes, batch_labtest_codes, \
                batch_diag_codes, batch_proc_codes, batch_labels, options)

        batch_med_codes = torch.LongTensor(batch_med_codes).to(options['device'])
        batch_diag_codes = torch.LongTensor(batch_diag_codes).to(options['device'])
        batch_mask_mult = torch.BoolTensor(1-batch_mask).unsqueeze(2).to(options['device'])     
        batch_mask_final = torch.Tensor(batch_mask_final).unsqueeze(2).to(options['device'])
        batch_med_mask_code = torch.Tensor(batch_med_mask_code).unsqueeze(3).to(options['device'])
        batch_diag_mask_code = torch.Tensor(batch_diag_mask_code).unsqueeze(3).to(options['device'])

        batch_s = torch.from_numpy(np.array(batchS)).float().to(options['device'])
        seq_time_step = torch.from_numpy(seq_time_step).float().to(options['device'])
        batch_global_time = torch.from_numpy(batch_global_time).float().to(options['device'])
        lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diag_codes])).to(options['device'])


        if options['dataset'] == 'mimic_data':
            batch_labtest_codes = torch.LongTensor(batch_labtest_codes).to(options['device'])
            batch_labtest_mask_code = torch.Tensor(batch_labtest_mask_code).unsqueeze(3).to(options['device'])
            batch_proc_mask_code = torch.Tensor(batch_proc_mask_code).unsqueeze(3).to(options['device'])
            batch_proc_codes = torch.LongTensor(batch_proc_codes).to(options['device'])
            # B, T, H
            features = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes,
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code, batch_proc_mask_code,
                    seq_time_step, lengths, options)

        elif options['dataset'] == 'eicu_data':
            batch_labtest_codes = torch.LongTensor(batch_labtest_codes).to(options['device'])
            batch_labtest_mask_code = torch.Tensor(batch_labtest_mask_code).unsqueeze(3).to(options['device'])
            features = self.feature_encoder(batch_med_codes, batch_labtest_codes, batch_diag_codes,
                batch_mask_mult, batch_med_mask_code, batch_labtest_mask_code, batch_diag_mask_code,
                    seq_time_step, lengths, options)

        elif options['dataset'] == 'DAPS_data':
            batch_proc_mask_code = torch.Tensor(batch_proc_mask_code).unsqueeze(3).to(options['device'])
            batch_proc_codes = torch.LongTensor(batch_proc_codes).to(options['device'])
            features = self.feature_encoder(batch_med_codes, batch_diag_codes, batch_proc_codes,
                    batch_mask_mult, batch_med_mask_code, batch_diag_mask_code, batch_proc_mask_code,
                        seq_time_step, lengths, options)

        final_statues = features * batch_mask_final
        final_statues = final_statues.sum(1, keepdim=True)  # B, 1, H
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, batch_mask_mult)    # B, T, 1

        time_weight = self.time_encoder(batch_global_time, quiryes, options, batch_mask_mult)       # B, T, 1

        attention_weight = torch.softmax(self.quiry_2weight_layer(final_statues), dim=2)    # B, 1, 2

        total_weight = torch.cat((time_weight, self_weight), dim=2)     # B, T, 2
        total_weight = torch.sum(total_weight*attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)

        weighted_features = features * total_weight     # B, T, H
        A = weighted_features
        B = features
        weighted_features = A * torch.sigmoid(B)
        averaged_features = torch.sum(weighted_features, dim=1) # B, H

        ## predict
        averaged_features = self.dropout(averaged_features)
        static_outputs = self.static_trans(batch_s)
        averaged_features = F.relu(static_outputs+averaged_features)
        predictions = self.classify_layer(averaged_features)
        predictions = torch.sigmoid(predictions)
        labels = torch.Tensor(batch_labels)
        labels = labels.to(options['device'])
        return predictions, labels, self_weight
