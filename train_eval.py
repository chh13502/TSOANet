import os
import time
import random
from models.utils import *
from models.model import *
from util import *
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import heapq
import operator



def train_model(seq_file='seq_file',
                label_file='label_file',
                time_file='time_file',
                n_med_codes=100000,
                n_labtest_codes=10000,
                n_diag_codes=10000,
                n_proc_codes=10000,
                n_labels=2,
                output_file='output_file',
                early_stop=10,
                min_epoch=0,
                batch_size=100,
                dp=0.5,
                lr=1e-4,
                L2_reg=0.001,
                n_epoch=1000,
                log_eps=1e-8,
                visit_size=512,
                hidden_size=256,
                use_gpu=False,
                model_name='',
                dataset = 'hf',
                running_data='',
                gamma=0.5,
                model_file = None,
                num_layers=1,
                ffn_dim=512,
                all_input_dim=2456,
                predDiag='',
                predLabtest='',
                predProc='',
                static_dim=113,
                model_choice='',
                k=0,
                ):

    device=torch.device("cuda:{}".format(use_gpu) if torch.cuda.is_available()==True else 'cpu')
    print("available device:{}\nload model...".format(device))
    options = locals().copy()

    model = model_file(n_med_codes, n_labtest_codes, n_diag_codes, n_proc_codes, batch_size, options).to(device)
    crossEntropy = CrossEntropy(n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = options['L2_reg'])

    print('loading data ...')
    trainSet, validSet, testSet = load_data(seq_file, label_file, time_file)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batch_size)))

    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    best_valid_auc = 0.
    best_test_auc = 0.
    epoch_duaration = 0.0
    best_epoch = 0.0
    max_len = 500000    # 想要取用的就诊记录次数（长度）
    best_parameters_file = ''
    model.to(device)
    # if use_gpu:
    #     model.cuda()
    model.train()
    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0

        for index in samples:
            batchX = trainSet[0][batch_size * index: batch_size * (index + 1)]
            batchY = trainSet[1][batch_size * index: batch_size * (index + 1)]
            batchS = trainSet[-1][batch_size * index: batch_size * (index + 1)]
            batchT = trainSet[2][batch_size * index: batch_size * (index + 1)]
            batchGT = trainSet[3][batch_size * index: batch_size * (index + 1)]

            batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, \
            batch_labels, batch_original_y = padInputWithTime_new(batchX, batchY, batchT, batchGT, options)
            batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time = adjust_input_new(batch_med_codes, \
                batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, max_len, n_med_codes, n_labtest_codes, \
                n_diag_codes, n_proc_codes, options)
            lengths = np.array([len(seq) for seq in batch_diag_codes])
            maxlen = np.max(lengths)
            predictions, labels, self_attention = model(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, batch_labels, options, maxlen, batchS)

            optimizer.zero_grad()

            # loss = focal_loss(predictions, labels, options)
            loss = crossEntropy(predictions, labels)
            loss.backward()
            optimizer.step()

            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 500 == 0):
                print('k:%d, epoch:%d, iteration:%d/%d, cost:%f' % (k, epoch, iteration, n_batches, loss.cpu().data.numpy()))
                #print(self_attention[:,0,0].squeeze().cpu().data.numpy())
                #print(time_weight[:, 0])
                #print(prior_weight[:, 0])
                #print(model.time_encoder.time_weight[0:10])
                #print(self_weight[:, 0])
            iteration += 1

        duration = time.time() - start_time
        if predDiag:
            print("======Diagnosis========")
        elif predLabtest:
            print("======Labtest=========")
        elif predProc:
            print("======Proc=========")
        else:
            print("======medic=======")
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        validate_cost, validate_auc, validate_aupr = calculate_cost_tran(model, validSet, options, max_len, crossEntropy)
        test_cost, test_auc, test_aupr = calculate_cost_tran(model, testSet, options, max_len, crossEntropy)
        print('k:%d, epoch:%d, validate_cost:%f, duration:%f' % (k, epoch, validate_cost, duration))
        epoch_duaration += duration

        if validate_auc > best_valid_auc:
            best_valid_auc = validate_auc
            best_test_auc = test_auc
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            best_parameters_file = os.path.join(output_file, f'model-{k}')
            torch.save(model.state_dict(), best_parameters_file)

        if epoch > min_epoch and epoch - best_epoch > early_stop:
            print(validate_cost)
            print(best_validate_cost)
            break
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f ,\n' \
              'Valid_auc:%.6f Test_auc:%.6f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost, best_valid_auc, best_test_auc)
        print(buf)
        print(model_name)
        with open(output_file + 'best_epoch_record.txt', 'a') as f:
            f.write(buf)
            f.write("\n{}\n".format(str(validate_auc)))
            f.write("\n")
    # testing
    model.load_state_dict(torch.load(best_parameters_file))
    model.eval()
    n_batches = int(np.ceil(float(len(testSet[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    v_trueVec = []
    v_predVec = []
    total_pred = []
    total_true = []
    for index in range(n_batches):
        batchX = testSet[0][batch_size * index: batch_size * (index + 1)]
        batchY = testSet[1][batch_size * index: batch_size * (index + 1)]
        batchS = testSet[-1][batch_size * index: batch_size * (index + 1)]
        batchT = testSet[2][batch_size * index: batch_size * (index + 1)]
        batchGT = testSet[3][batch_size * index: batch_size * (index + 1)]

        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time,\
            batch_labels, batch_original_y = padInputWithTime_new(batchX, batchY, batchT, batchGT, options)
        batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time = adjust_input_new(batch_med_codes, \
            batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, max_len, n_med_codes, n_labtest_codes, \
            n_diag_codes, n_proc_codes, options)
        lengths = np.array([len(seq) for seq in batch_diag_codes])
        maxlen = np.max(lengths)
        predictions, labels, self_attention = model(batch_med_codes, batch_labtest_codes, batch_diag_codes, batch_proc_codes, batch_time_step, batch_global_time, batch_labels, options, maxlen, batchS)


        pred_score = predictions.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        for i in range(predictions.shape[0]):
            tensorMatrix = pred_score[i,:]
            # tensorMatrix = pad_prev_y[i,:,:]
            thisY = batch_original_y[i]
            all_y = y_true[i,:]
            if len(thisY) == 0: continue
            v_trueVec.append(thisY)
            output = tensorMatrix
            v_predVec.append(list(zip(*heapq.nlargest(50, enumerate(output), key=operator.itemgetter(1))))[0])
            total_pred.append(tensorMatrix)
            total_true.append(all_y)
    
    recall = recallTop(v_trueVec, v_predVec, rank=[10,20,30,40,50])
    precision = precisionTop(v_trueVec, v_predVec, rank=[10,20,30,40,50])
    F1 = F1Top(v_trueVec, v_predVec, rank=[10,20,30,40,50])
    print('test=======>>\nrecall@10:{}, recall@20:{}, recall@30:{} recall@40:{} recall@50:{}'.format(recall[0], recall[1], recall[2], recall[3], recall[4]))
    print('test----->>\nprecision@10:{}, precision@20:{}, precision@30:{}, precision@40:{}, precision@50:{}'.format(precision[0], \
            precision[1], precision[2], precision[3], precision[4]))
    print('===>>F1@10:{}, F1@20:{}. f1@30:{}, f1@40:{}, f1@50:{}'.format(F1[0], F1[1], F1[2], F1[3], F1[4]))
    total_pred_value = np.concatenate(total_pred)
    total_true_value = np.concatenate(total_true)
    print("======>>>>",total_pred_value.shape, total_true_value.shape)
    total_avg_auc_micro = roc_auc_score(total_true_value, total_pred_value, average='micro')
    total_avg_auc_macro = roc_auc_score(total_true_value, total_pred_value, average='macro')
    total_avg_aupr_micro = average_precision_score(total_true_value, total_pred_value, average='micro')
    total_avg_aupr_macro = average_precision_score(total_true_value, total_pred_value, average='macro')
    print("micro=====auc:{}\taupr:{}\nmacro----->>auc:{}\taupr:{}".format(
        total_avg_auc_micro, total_avg_aupr_micro, total_avg_auc_macro, total_avg_aupr_macro
    ))

    return precision[0], precision[1], precision[2], precision[3], precision[4], recall[0], recall[1], recall[2], recall[3], recall[4], \
            F1[0], F1[1], F1[2], F1[3], F1[4], total_avg_auc_micro, total_avg_aupr_micro

def parse_arguments(parser):
    parser.add_argument('--seed', type=int, default=43, help='random seed(default value is 43)')
    parser.add_argument('--dataset', type=str, default='mimic_data')
    parser.add_argument('--num_layers', type=int, default=2, help='number of EncoderLayers')
    parser.add_argument('--num_models', type=int, default=10, help='number of models')
    parser.add_argument('--ffn_dim', type=int, default=1024, help='# dimension of feed fordward network')
    parser.add_argument('--visit_size', type=int, default=256, help='The main input embedding dimersion(default value is 200)')
    parser.add_argument('--hidden_size', type=int, default=256, help='The lstm hidden dimersion(default is 200)')
    parser.add_argument('--static_dim', type=int, default=113, help='The static info dimersion.(default is 113)')
    parser.add_argument('--L2_output', type=float, default=0.001, help='The L2 regularization for the output layer(default is 0.001)')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='The dropout rate(default is 0.5)')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate(default is 1e-3)')
    parser.add_argument('--bs', type=int, default=10, help='The batch size of input(default is 10)')
    parser.add_argument('--max_epoch', type=int, default=50, help='The max epoch to train(default is 200)')
    parser.add_argument('--min_epoch', type=int, default=0, help='The min epoch to train(default is 0)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop epochs.')
    parser.add_argument('--gpu', help='use gpu to train.')
    parser.add_argument('--predDiag', type=int, default=0, help='Predict diagnosis code')
    parser.add_argument('--predProc', type=int, default=0, help='Predict procedure code')
    parser.add_argument('--predLabtest', type=int, default=0, help='Predict labtest code')
    parser.add_argument('--predMedic', type=int, default=0, help='Predict labtest code')
    parser.add_argument('--model_choice', type=str, default='TSOANet')
    args = parser.parse_args()
    return args

# For real data set please contact the corresponding author

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    num_models = args.num_models
    batch_size = args.bs
    seed = args.seed
    dropout_rate = args.dropout_rate
    lr = args.lr
    L2_reg = args.L2_output
    log_eps = 1e-8
    n_epoch = args.max_epoch
    visit_size = args.visit_size    # size of input embedding
    hidden_size = args.hidden_size  # size of hidden layer
    num_layers = args.num_layers    # layer of Transformer
    ffn_dim = args.ffn_dim
    gamma = 2       # setting for Focal Loss, when it's zero, it's equal to standard cross loss
    use_gpu = args.gpu
    model_choice = args.model_choice
    early_stop = args.early_stop
    min_epoch = args.min_epoch

    model_file = eval(model_choice)
    dataset = args.dataset

    model_name = 'tran_%s_%s_L%d_ffn%d_lr%f_dp%f_vs%d_hs%d_focal%.2f_bs%d' % (model_choice, dataset, num_layers, ffn_dim, lr, dropout_rate, visit_size, hidden_size, gamma, batch_size)
    if dataset == 'mimic_data':
        seq_file="./data/mimic_data/seqFile/"
        label_file="./data/mimic_data/labelFile/"
        time_file="./data/mimic_data/timeFile/"
        out_file=f"./result/MIMIC_result/"
        inputDim = 2456
        medDim = 346
        labtestDim = 676
        diagDim = 905
        procDim = 529
        static_dim = 113
        if args.predDiag:
            n_labels = 905
            output_file_path = os.path.join(out_file, model_choice, "predDiag", model_name) + "/"
        elif args.predProc:
            n_labels = 529
            output_file_path = os.path.join(out_file, model_choice, "predProc", model_name) + "/"
        elif args.predLabtest:
            n_labels = 676
            output_file_path = os.path.join(out_file, model_choice, "predLabtest", model_name) + "/"
        else:
            n_labels = 346
            output_file_path = os.path.join(out_file, model_choice, "predMedic", model_name) + "/"
    if dataset == 'eicu_data':
        if args.predProc:
            raise ValueError
        seq_file = "data/eicu_data/seqFile/"
        label_file = "data/eicu_data/labelFile/"
        time_file = "data/eicu_data/timeFile/"
        out_file ="./result/eICU_result/"
        inputDim=2895
        medDim = 1374
        labtestDim = 155
        diagDim = 1366
        procDim = 0
        static_dim=12
        if args.predDiag:
            n_labels = 1366
            output_file_path = os.path.join(out_file, model_choice, "predDiag", model_name) + "/"
        elif args.predLabtest:
            n_labels = 155
            output_file_path = os.path.join(out_file, model_choice, "predLabtest", model_name) + "/"
        else:
            n_labels = 1374
            output_file_path = os.path.join(out_file, model_choice, "predMedic", model_name) + "/"
    if dataset == 'DAPS_data':
        if args.predLabtest:
            raise ValueError
        seq_file = "data/DAPS_data/seqFile/"
        label_file = "data/DAPS_data/labelFile/"
        time_file = "data/DAPS_data/timeFile/"
        out_file ="./result/DAPS_data/"
        inputDim = 0
        medDim = 132
        labtestDim = 0
        diagDim = 1958
        procDim = 1430
        static_dim = 10
        if args.predDiag:
            n_labels = 1958
            output_file_path = os.path.join(out_file, model_choice, "predDiag", model_name) + "/"
        elif args.predProc:
            n_labels = 1430
            output_file_path = os.path.join(out_file, model_choice, "predProc", model_name) + "/"
        else:
            n_labels = 131
            output_file_path = os.path.join(out_file, model_choice, "predMedic", model_name) + "/"
    print(model_name)

    output_file_path = output_file_path.replace('\\', '/')

    log_file = output_file_path
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    code2id = None
    # n_diag_codes = inputDim
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)
        
    results = []
    print("======>>>dataset:{}".format(dataset))
    set_seed(seed)
    for k in range(num_models):
        precision10, precision20, precision30, precision40, precision50,  \
        recall10, recall20, recall30, reacall40, recall50, \
        f1_10, f1_20, f1_30, f1_40, f1_50, \
        roc_auc, aupr = train_model(seq_file, label_file,
                                time_file, medDim, labtestDim, diagDim, procDim, n_labels,
                                output_file_path, early_stop, min_epoch, batch_size, dropout_rate, lr,
                                L2_reg, n_epoch, log_eps, visit_size, hidden_size,
                                use_gpu, model_name, dataset=dataset,
                                gamma=gamma, num_layers=num_layers, ffn_dim=ffn_dim, model_file=model_file, \
                                all_input_dim=inputDim, predDiag=args.predDiag, \
                                predLabtest= args.predLabtest, predProc=args.predProc, \
                                static_dim=static_dim, model_choice=model_choice, k=k)
        results.append([precision10, precision20, precision30, precision40, precision50,\
                        recall10, recall20, recall30, reacall40, recall50, f1_10, f1_20, f1_30, f1_40, f1_50, roc_auc, aupr])

    results = np.array(results)
    print(np.mean(results, 0))
    print(np.std(results, 0))
    with open(log_file + 'result.txt', 'a') as f:
        f.write(model_name)
        f.write('\n')
        all_result_mean = np.mean(results, 0)
        all_result_std = np.std(results, 0)
        f.write("precision10:{:6f}+{:6f}\nprecision20:{:6f}+{:6f}\nprecision30:{:6f}+{:6f}\nprecision40:{:6f}+{:6f}\nprecision50:{:6f}+{:6f}\n".format(
            all_result_mean[0],all_result_std[0], \
            all_result_mean[1],all_result_std[1], \
            all_result_mean[2],all_result_std[2], \
            all_result_mean[3],all_result_std[3], \
            all_result_mean[4],all_result_std[4]))

        f.write("f1_10:{:6f}+{:6f}\nf1_20:{:6f}+{:6f}\nf1_30:{:6f}+{:6f}\nf1_40:{:6f}+{:6f}\nf1_50:{:6f}+{:6f}\n".format(
            all_result_mean[10],all_result_std[10], \
            all_result_mean[11],all_result_std[11], \
            all_result_mean[12],all_result_std[12], \
            all_result_mean[13],all_result_std[13], \
            all_result_mean[14],all_result_std[14]
            ))
        f.write("recall10:{:6f}+{:6f}\nreacall20:{:6f}+{:6f}\nrecall30:{:6f}+{:6f}\nracall40:{:6f}+{:6f}\nracall50:{:6f}+{:6f}\n".format(
            all_result_mean[5],all_result_std[5], \
            all_result_mean[6],all_result_std[6], \
            all_result_mean[7],all_result_std[7], \
            all_result_mean[8],all_result_std[8], \
            all_result_mean[9],all_result_std[9]
            ))
        f.write("auc:{:6f}+{:6f}\taupr:{:6f}+{:6f}\n".format(
            all_result_mean[15],all_result_std[15], \
            all_result_mean[16],all_result_std[16]
            ))
        f.write('\n')
        f.write(str(np.std(results, 0)))
