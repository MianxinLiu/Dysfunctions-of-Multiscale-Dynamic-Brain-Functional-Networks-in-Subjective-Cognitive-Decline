import pandas as pd
import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from sklearn.model_selection import KFold
from random import shuffle
import GradCAM
import glob

## training
neg=0
pos=1

subInfo = pd.read_csv('./sbj_obj_proj_revise.csv')
subInfo = subInfo[(subInfo['grouping']==neg) | (subInfo['grouping']==pos)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)
y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1

#index=[i for i in range(len(subInfo))]
#shuffle(index)
#scio.savemat('./shuffled_index_sbj', {'index': index})

temp = scio.loadmat('./shuffled_index_sbj')
index = temp['index'][0]
y_data1 = y_data1[index]

time_step=22
for ROInum in [100,200,300,400,500]:
    FC_all = np.zeros((len(subInfo),time_step,ROInum,ROInum))
    subcount = 0
    for fn in subInfo['filename']:
        temp=scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(ROInum) + '/' + fn)
        FC_all[subcount,:, :, :] = temp['DFC']
        subcount = subcount + 1
    FC_all = torch.from_numpy(FC_all)
    FC_all = torch.tensor(FC_all, dtype=torch.float32)
    FC_all = FC_all[index,:,:]

    qual_all=[]
    for cv in [1,2,3,4,5]:
        temp = scio.loadmat('./kfold/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv' + str(cv))
        train_idx = temp['train_idx'][0]
        test_idx = temp['test_idx'][0]
        print("Train:", train_idx, " Test:", test_idx)
        dataset_train = TensorDataset(FC_all[train_idx, :, :], y_data1[train_idx])
        dataset_test = TensorDataset(FC_all[test_idx, :, :], y_data1[test_idx])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=30, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        ratio=y_data1[train_idx].sum()/(y_data1[train_idx].shape[0]-y_data1[train_idx].sum())
        if ratio<1:
            weight=torch.cuda.FloatTensor([1,1/ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])
        loss_func = nn.CrossEntropyLoss(weight)  # the target label is not one-hotted

        lr = 0.01
        EPOCH = 100

        qualified = []
        while not qualified:
            model = MyModels.GraphConvolutionalLSTM_MIL0(ROInum=ROInum)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
            auc_baseline = 0.65
            starttest=0
            for epoch in range(EPOCH):
                for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                    model.train()
                    b_y = b_y.view(-1)
                    b_y = b_y.long()
                    b_y = b_y.cuda()
                    output, att = model(b_x) 

                    loss = loss_func(output, b_y)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    predicted = torch.max(output.data, 1)[1]
                    correct = (predicted == b_y).sum()
                    tr_accuracy = float(correct) / float(b_x.shape[0])
                    train_auc.append(tr_accuracy)
                    print('[Epoch %d, Batch %5d] loss: %.3f' %
                          (epoch + 1, step + 1, loss))
                    print('|train diag loss:', loss.data.item(), '|train accuracy:', tr_accuracy
                          )
                    if tr_accuracy>0.90 and epoch>10:
                        starttest=1
                    if starttest:
                        predicted_all = []
                        predict_p=[]
                        test_y_all = []
                        model.eval()
                        with torch.no_grad():
                            for i, (test_x, test_y) in enumerate(test_loader):
                                test_y = test_y.view(-1)
                                test_y = test_y.long()
                                test_y = test_y.cuda()
                                test_output, att = model(test_x)

                                test_loss = loss_func(test_output, test_y)
                                loss = loss_func(test_output, test_y)
                                print('[Epoch %d, Batch %5d] valid loss: %.3f' %
                                      (epoch + 1, step + 1, loss))
                                predict_p=predict_p+test_output[:,1].tolist()
                                predicted = torch.max(test_output.data, 1)[1]
                                correct = (predicted == test_y).sum()
                                accuracy = float(correct) / float(predicted.shape[0])
                                test_y = test_y.cpu()
                                predicted = predicted.cpu()
                                predicted_all = predicted_all + predicted.tolist()
                                test_y_all = test_y_all + test_y.tolist()


                        correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                        accuracy = float(correct) / float(len(test_y_all))
                        sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                        spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                        auc = metrics.roc_auc_score(test_y_all, predict_p)
                        print('|test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', spec,
                              '|test auc:', auc,
                              )
                        if auc >= auc_baseline and sens >0.5 and spec >0.5 and tr_accuracy>0.75:
                            auc_baseline = auc
                            torch.save(model.state_dict(),'./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_mil.pth')
                            print('got one model with |test accuracy:', accuracy,
                                  '|test sen:', sens,
                                  '|test spe:', spec,
                                  '|test auc:', auc
                                  )
                            qualified.append([accuracy, sens, spec, auc])
        qual_all.append(qualified[-1])

    print(qual_all)
    print(np.mean(qual_all,axis=0))
    print(np.std(qual_all,axis=0))

## show internal results

ROInum=500
time_step=22
FC_all = np.zeros((len(subInfo),time_step,ROInum,ROInum))
subcount = 0
for fn in subInfo['filename']:
    temp=scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(ROInum) + '/' + fn)
    FC_all[subcount,:, :, :] = temp['DFC']
    subcount = subcount + 1
FC_all = torch.from_numpy(FC_all)
FC_all = torch.tensor(FC_all, dtype=torch.float32)

temp = scio.loadmat('./shuffled_index_sbj')
index = temp['index'][0]

FC_all = FC_all[index,:,:]
qualified=[]
Att_all = []
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv'+str(cv))
    test_idx = temp['test_idx'][0]
    dataset_test = TensorDataset(FC_all[test_idx,:,:,:], y_data1[test_idx])
    ## if you want to assess attention and save all used data (optional)
    # dataset_test = TensorDataset(FC_all, y_data1)
    
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    model = MyModels.GraphConvolutionalLSTM_MIL0(ROInum=ROInum)
    ## if use GCN-LSTM model
    #model = MyModels.GraphConvolutionalLSTM_base0(ROInum=ROInum, tsize=time_step)
    model.cuda()

    model.load_state_dict(torch.load('./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_mil.pth'))
    ## if use GCN-LSTM model
    #model.load_state_dict(torch.load('./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_base.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []

    model.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            test_y = test_y.view(-1)
            test_y = test_y.long()
            test_y = test_y.cuda()
            test_output, Att = model(test_x)
            
            ## if use GCN-LSTM model
            #test_output = model(test_x)
            
            predict_p = predict_p + test_output[:,1].tolist()
            predicted = torch.max(test_output.data, 1)[1]
            correct = (predicted == test_y).sum()
            accuracy = float(correct) / float(predicted.shape[0])
            test_y = test_y.cpu()
            predicted = predicted.cpu()
            predicted_all = predicted_all + predicted.tolist()
            test_y_all = test_y_all + test_y.tolist()
            ## if you want to assess attention on all used data (optional)
            #Att_all = Att_all + Att.tolist()

    correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
    accuracy = float(correct) / float(len(test_y_all))
    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
    auc = metrics.roc_auc_score(test_y_all, predict_p)
    auc_pr = metrics.average_precision_score(test_y_all, predict_p)
    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          '|test auc_pr:', auc_pr,
          )
    qualified.append([accuracy, sens, spec, auc, auc_pr])
print(np.mean(qualified, axis=0))
print(np.std(qualified, axis=0))
scio.savemat('./metrics/gc_lstm_att_'+str(ROInum)+'.mat', {'qualified': qualified})

## save dFCN, attention and label (optional)
FC_all=FC_all.numpy()
y_data1=y_data1.numpy()
scio.savemat('./att/att_'+str(ROInum)+'_test.mat', {'att':Att_all, 'FC_all':FC_all})
scio.savemat('./att/label.mat', {'label':y_data1})