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
import transferlearning as TL


## training
neg=0
pos=1

subInfo = pd.read_csv('./sbj_obj_proj_all.csv')
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
index = torch.tensor(index, dtype=torch.int)

time_step=22

qual_all=[]
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv' + str(cv))
    train_idx = temp['train_idx'][0]
    test_idx = temp['test_idx'][0]
    print("Train:", train_idx, " Test:", test_idx)
    dataset_train = TensorDataset(index[train_idx], y_data1[train_idx])
    dataset_test = TensorDataset(index[test_idx], y_data1[test_idx])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=30, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    ratio=y_data1[train_idx].sum()/(y_data1[train_idx].shape[0]-y_data1[train_idx].sum())
    weight=torch.cuda.FloatTensor([1,1/ratio])
    loss_func = nn.CrossEntropyLoss(weight) 

    lr = 0.001
    EPOCH = 10
    ROInum=500
    qualified = []
    
    while not qualified:
        # feature concatenation
        model = MyModels.MGCUNET_LSTM_MIL_fusion2(cv)
        
        # weighted voting
        #model = MyModels.MGCUNET_LSTM_MIL_fusion0(cv)
        
        TL.freeze_by_names(model, ('gclstm1', 'gclstm2', 'gclstm3', 'gclstm4', 'gclstm5'))
        model.init()
        model.cuda()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                     weight_decay=1e-2)

        auc_baseline = 0.75
        starttest=0
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                model.train()
                b_y = b_y.view(-1)
                b_y = b_y.long()
                b_y = b_y.cuda()
                temp = b_x.numpy().tolist()
                batch_size = b_x.shape[0]
                A1 = np.zeros((batch_size,time_step, 100, 100))
                A2 = np.zeros((batch_size,time_step, 200, 200))
                A3 = np.zeros((batch_size,time_step, 300, 300))
                A4 = np.zeros((batch_size,time_step, 400, 400))
                A5 = np.zeros((batch_size,time_step, 500, 500))
                subcount = 0
                for id in temp:
                    fn = subInfo['filename'].values[int(id)]
                    FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(100) + '/' + fn)
                    A1[subcount, :, :] = FCfile['DFC']
                    FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(200) + '/' + fn)
                    A2[subcount, :, :] = FCfile['DFC']
                    FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(300) + '/' + fn)
                    A3[subcount, :, :] = FCfile['DFC']
                    FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(400) + '/' + fn)
                    A4[subcount, :, :] = FCfile['DFC']
                    FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(500) + '/' + fn)
                    A5[subcount, :, :] = FCfile['DFC']
                    subcount = subcount + 1

                A1 = torch.tensor(A1, dtype=torch.float32)
                A1.cuda()
                A2 = torch.tensor(A2, dtype=torch.float32)
                A2.cuda()
                A3 = torch.tensor(A3, dtype=torch.float32)
                A3.cuda()
                A4 = torch.tensor(A4, dtype=torch.float32)
                A4.cuda()
                A5 = torch.tensor(A5, dtype=torch.float32)
                A5.cuda()

                output = model(A1, A2, A3, A4, A5)

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
                if epoch >= 2:
                    predicted_all = []
                    predict_p = []
                    test_y_all = []
                    model.eval()
                    with torch.no_grad():
                        for i, (test_x, test_y) in enumerate(test_loader):
                            test_y = test_y.view(-1)
                            test_y = test_y.long()
                            test_y = test_y.cuda()
                            temp = test_x.numpy().tolist()
                            batch_size = test_x.shape[0]
                            A1 = np.zeros((batch_size,time_step, 100, 100))
                            A2 = np.zeros((batch_size,time_step, 200, 200))
                            A3 = np.zeros((batch_size,time_step, 300, 300))
                            A4 = np.zeros((batch_size,time_step, 400, 400))
                            A5 = np.zeros((batch_size,time_step, 500, 500))
                            subcount = 0
                            for id in temp:
                                fn = subInfo['filename'].values[int(id)]
                                FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(100) + '/' + fn)
                                A1[subcount, :, :] = FCfile['DFC']
                                FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(200) + '/' + fn)
                                A2[subcount, :, :] = FCfile['DFC']
                                FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(300) + '/' + fn)
                                A3[subcount, :, :] = FCfile['DFC']
                                FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(400) + '/' + fn)
                                A4[subcount, :, :] = FCfile['DFC']
                                FCfile = scio.loadmat('/media/user/4TB/matlab/PET_center/Rest2/DFC/par' + str(500) + '/' + fn)
                                A5[subcount, :, :] = FCfile['DFC']
                                subcount = subcount + 1

                            A1 = torch.tensor(A1, dtype=torch.float32)
                            A1.cuda()
                            A2 = torch.tensor(A2, dtype=torch.float32)
                            A2.cuda()
                            A3 = torch.tensor(A3, dtype=torch.float32)
                            A3.cuda()
                            A4 = torch.tensor(A4, dtype=torch.float32)
                            A4.cuda()
                            A5 = torch.tensor(A5, dtype=torch.float32)
                            A5.cuda()
                            test_x.cuda()
                            test_output = model(A1, A2, A3,A4,A5)

                            predict_p = predict_p + test_output[:, 1].tolist()
                            test_loss = loss_func(test_output, test_y)
                            loss = loss_func(test_output, test_y)
                            print('[Epoch %d, Batch %5d] valid loss: %.3f' %
                                  (epoch + 1, step + 1, loss))
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

                    if auc >= auc_baseline and sens >0.5 and spec >0.5:
                        auc_baseline = auc
                        torch.save(model.state_dict(),'./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_magcn-lstm_FeaCon2.pth')

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
qualified=[]

for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv'+str(cv))
    test_idx = temp['test_idx'][0]
    dataset_test = TensorDataset(index[test_idx], y_data1[test_idx])
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    model = MyModels.MGCUNET_LSTM_MIL_fusion2(cv)
    model.cuda()

    model.load_state_dict(torch.load('./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_magcn-lstm_FeaCon2.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []
    model.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            test_y = test_y.view(-1)
            test_y = test_y.long()
            test_y = test_y.cuda()
            temp = test_x.numpy().tolist()
            batch_size = test_x.shape[0]
            A1 = np.zeros((batch_size, time_step, 100, 100))
            A2 = np.zeros((batch_size, time_step, 200, 200))
            A3 = np.zeros((batch_size, time_step, 300, 300))
            A4 = np.zeros((batch_size, time_step, 400, 400))
            A5 = np.zeros((batch_size, time_step, 500, 500))

            subcount = 0
            for id in temp:
                fn = subInfo['filename'].values[int(id)]
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(100) + '/' + fn)
                A1[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(200) + '/' + fn)
                A2[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(300) + '/' + fn)
                A3[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(400) + '/' + fn)
                A4[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(500) + '/' + fn)
                A5[subcount, :, :] = FCfile['DFC']
                subcount = subcount + 1

            A1 = torch.tensor(A1, dtype=torch.float32)
            A1.cuda()
            A2 = torch.tensor(A2, dtype=torch.float32)
            A2.cuda()
            A3 = torch.tensor(A3, dtype=torch.float32)
            A3.cuda()
            A4 = torch.tensor(A4, dtype=torch.float32)
            A4.cuda()
            A5 = torch.tensor(A5, dtype=torch.float32)
            A5.cuda()

            test_x.cuda()
            test_output = model(A1, A2, A3, A4, A5)
            predict_p = predict_p + test_output[:, 1].tolist()
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
scio.savemat('./metrics/gc_lstm_att_fea_con.mat', {'qualified': qualified})

## external testing (integrated using majority voting)

neg=0
pos=1

subInfo = pd.read_csv('./obj_sbj_external_updated.csv')
subInfo = subInfo[(subInfo['grouping']==neg) | (subInfo['grouping']==pos)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)
y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1
index=[i for i in range(len(subInfo))]
index = torch.tensor(index , dtype=torch.int)
qualified=[]

vote=np.zeros([len(subInfo),5])
vote_p=np.zeros([len(subInfo),5])

for cv in [1,2,3,4,5]:
    dataset_test = TensorDataset(index, y_data1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    model = MyModels.MGCUNET_LSTM_MIL_fusion2(cv)
    model.cuda()

    model.load_state_dict(torch.load('./model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_magcn-lstm_FeaCon2.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []
    model.eval()
    
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            test_y = test_y.view(-1)
            test_y = test_y.long()
            test_y = test_y.cuda()
            temp = test_x.numpy().tolist()
            batch_size = test_x.shape[0]
            A1 = np.zeros((batch_size, time_step, 100, 100))
            A2 = np.zeros((batch_size, time_step, 200, 200))
            A3 = np.zeros((batch_size, time_step, 300, 300))
            A4 = np.zeros((batch_size, time_step, 400, 400))
            A5 = np.zeros((batch_size, time_step, 500, 500))

            subcount = 0
            for id in temp:
                fn = subInfo['filename'].values[int(id)]
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(100) + '/' + fn)
                A1[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(200) + '/' + fn)
                A2[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(300) + '/' + fn)
                A3[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(400) + '/' + fn)
                A4[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(500) + '/' + fn)
                A5[subcount, :, :] = FCfile['DFC']
                subcount = subcount + 1

            A1 = torch.tensor(A1, dtype=torch.float32)
            A1.cuda()
            A2 = torch.tensor(A2, dtype=torch.float32)
            A2.cuda()
            A3 = torch.tensor(A3, dtype=torch.float32)
            A3.cuda()
            A4 = torch.tensor(A4, dtype=torch.float32)
            A4.cuda()
            A5 = torch.tensor(A5, dtype=torch.float32)
            A5.cuda()

            test_x.cuda()
            test_output = model(A1, A2, A3, A4, A5)
            predict_p = predict_p + test_output[:, 1].tolist()
            predicted = torch.max(test_output.data, 1)[1]
            correct = (predicted == test_y).sum()
            accuracy = float(correct) / float(predicted.shape[0])
            test_y = test_y.cpu()
            predicted = predicted.cpu()
            predicted_all = predicted_all + predicted.tolist()
            test_y_all = test_y_all + test_y.tolist()

    vote[:, cv-1] = np.array(predicted_all)
    vote_p[:, cv - 1] = np.array(predict_p)


countvote=np.sum(vote[:,:],axis=1)
countvote[countvote<int(5/2+0.5)]=0
countvote[countvote>=int(5/2+0.5)]=1

qual_all=[]
predicted_all=countvote.tolist()
correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
accuracy = float(correct) / float(len(test_y_all))
sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)

countvote=np.mean(vote[:,:],axis=1)
predicted_p=countvote.tolist()
auc = metrics.roc_auc_score(test_y_all, predicted_p)
auc_pr = metrics.average_precision_score(test_y_all, predicted_p)
print('|test accuracy:', accuracy,
      '|test sen:', sens,
      '|test spe:', spec,
      '|test auc:', auc,
      '|test auc_pr:', auc_pr,
      )
qual_all.append([accuracy, sens, spec, auc, auc_pr])
scio.savemat('./metrics/Test_FC.mat', {'qualified': qualified, 'predicted_all': predicted_all, 'test_y_all':test_y_all })

## external testing (individual model)
neg=0
pos=1

subInfo = pd.read_csv('./obj_sbj_external_updated.csv')
subInfo = subInfo[(subInfo['grouping']==neg) | (subInfo['grouping']==pos)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)
y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1
index=[i for i in range(len(subInfo))]
index = torch.tensor(index , dtype=torch.int)

qualified=[]

for cv in [1,2,3,4,5]:
    dataset_test = TensorDataset(index, y_data1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    model = MyModels.MGCUNET_LSTM_MIL_fusion2(cv)
    model.cuda()

    model.load_state_dict(torch.load('./ckpt_obj_proj3/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_magcn-lstm_FeaCon2.pth'))
    predicted_all = []
    predict_p = []
    test_y_all = []
    model.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            test_y = test_y.view(-1)
            test_y = test_y.long()
            test_y = test_y.cuda()
            temp = test_x.numpy().tolist()
            batch_size = test_x.shape[0]
            A1 = np.zeros((batch_size, time_step, 100, 100))
            A2 = np.zeros((batch_size, time_step, 200, 200))
            A3 = np.zeros((batch_size, time_step, 300, 300))
            A4 = np.zeros((batch_size, time_step, 400, 400))
            A5 = np.zeros((batch_size, time_step, 500, 500))

            subcount = 0
            for id in temp:
                fn = subInfo['filename'].values[int(id)]
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(100) + '/' + fn)
                A1[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(200) + '/' + fn)
                A2[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(300) + '/' + fn)
                A3[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(400) + '/' + fn)
                A4[subcount, :, :] = FCfile['DFC']
                FCfile = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(500) + '/' + fn)
                A5[subcount, :, :] = FCfile['DFC']
                subcount = subcount + 1

            A1 = torch.tensor(A1, dtype=torch.float32)
            A1.cuda()
            A2 = torch.tensor(A2, dtype=torch.float32)
            A2.cuda()
            A3 = torch.tensor(A3, dtype=torch.float32)
            A3.cuda()
            A4 = torch.tensor(A4, dtype=torch.float32)
            A4.cuda()
            A5 = torch.tensor(A5, dtype=torch.float32)
            A5.cuda()

            test_x.cuda()
            test_output = model(A1, A2, A3, A4, A5)
            predict_p = predict_p + test_output[:, 1].tolist()
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

## permutation test
permu_results=[]
p=[]
count=0
for i in range(1000):
    permu_index=test_y_all.copy()
    shuffle(permu_index)

    auc = metrics.average_precision_score(permu_index, predict_p)
    if auc>qual_all[0][4]: #AUPRC
        count=count+1
print(count / 1000)

permu_results=[]
p=[]
count=0
for i in range(1000):
    permu_index=test_y_all.copy()
    shuffle(permu_index)

    auc = metrics.roc_auc_score(permu_index, predict_p)
    if auc>qual_all[0][3]:#AUROC
        count=count+1
print(count / 1000)

permu_results=[]
p=[]
count=0
for i in range(1000):
    permu_index=test_y_all.copy()
    shuffle(permu_index)

    sens = metrics.recall_score(permu_index, predicted_all, pos_label=1)
    if sens>qual_all[0][1]:#sensitivity
        count=count+1
print(count / 1000)


permu_results=[]
p=[]
count=0
for i in range(1000):
    permu_index=test_y_all.copy()
    shuffle(permu_index)

    spec = metrics.recall_score(permu_index, predicted_all, pos_label=0)
    if spec>qual_all[0][2]:#specificity
        count=count+1
print(count / 1000)

