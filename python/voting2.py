import pandas as pd
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from random import shuffle

## there is no training for this fusion method

## show internal results
neg=0
pos=1

subInfo = pd.read_csv('./sbj_obj_proj_revise.csv')
subInfo = subInfo[(subInfo['grouping']==neg) | (subInfo['grouping']==pos)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)
y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1

temp = scio.loadmat('./shuffled_index_sbj')
index = temp['index'][0]
y_data1 = y_data1[index]

ROI = 500
performance=np.zeros((5,5))
qualified=[]
for cv in [1,2,3,4,5]:
    temp = scio.loadmat('./kfold/' + str(neg) + 'vs' + str(pos) + '/shuffled_index_cv' + str(cv))
    test_idx = temp['test_idx'][0]
    vote=np.zeros([len(test_idx),int(ROI/100)])
    time_step = 22
    for ROInum in range(100,ROI+100,100):

        FC_all = np.zeros((len(subInfo), time_step, ROInum, ROInum))
        subcount = 0
        for fn in subInfo['filename']:
            temp = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(ROInum) + '/' + fn)
            FC_all[subcount, :, :, :] = temp['DFC']
            subcount = subcount + 1
        FC_all = torch.from_numpy(FC_all)
        FC_all = torch.tensor(FC_all, dtype=torch.float32)

        FC_all = FC_all[index, :, :]

        dataset_test = TensorDataset(FC_all[test_idx, :, :], y_data1[test_idx])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        model = MyModels.GraphConvolutionalLSTM_MIL0(ROInum=ROInum)
        model.load_state_dict(torch.load(
            './model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_mil.pth'))
        model.cuda()
        predicted_all = []
        predict_p = []
        test_y_all = []
        model.eval()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(test_loader):
                test_y = test_y.view(-1)
                test_y = test_y.long()
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output, att = model(test_x)
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

        vote[:,int((ROInum/100))-1]=np.array(predicted_all)

    predict_p=np.mean(vote,axis=1)
    predicted_all=predict_p.copy()
    predicted_all[predicted_all>=0.5]=1
    predicted_all[predicted_all<0.5]=0
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
    performance[cv - 1, 0] = accuracy
    performance[cv - 1, 1] = sens
    performance[cv - 1, 2] = spec
    performance[cv - 1, 3] = auc
    performance[cv - 1, 4] = auc_pr

print(np.mean(qualified, axis=0))
print(np.std(qualified, axis=0))
scio.savemat('./metrics/gc_lstm_att_MV.mat', {'qualified': qualified})

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

ROI =500
performance=np.zeros((5,4))
qualified=[]
vote_fold=np.zeros([len(subInfo),5])
vote_p=np.zeros([len(subInfo),5])
for cv in [1,2,3,4,5]:
    vote=np.zeros([len(subInfo),int(ROI/100)])
    time_step = 22
    for ROInum in range(100,ROI+100,100):
        FC_all = np.zeros((len(subInfo), time_step, ROInum, ROInum))
        subcount = 0
        for fn in subInfo['filename']:
            temp = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(ROInum) + '/' + fn)
            FC_all[subcount, :, :, :] = temp['DFC']
            subcount = subcount + 1
        FC_all = torch.from_numpy(FC_all)
        FC_all = torch.tensor(FC_all, dtype=torch.float32)

        dataset_test = TensorDataset(FC_all, y_data1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        model = MyModels.GraphConvolutionalLSTM_MIL0(ROInum=ROInum)
        model.load_state_dict(torch.load(
            './model_DFC/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_mil.pth'))
        model.cuda()
        predicted_all = []
        predict_p = []
        test_y_all = []
        model.eval()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(test_loader):
                test_y = test_y.view(-1)
                test_y = test_y.long()
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output, att = model(test_x)
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
        vote[:,int((ROInum/100))-1]=np.array(predicted_all)

    countvote=np.sum(vote[:,:],axis=1)
    countvote[countvote<int(ROI/100/2+0.5)]=0
    countvote[countvote>=int(ROI/100/2+0.5)]=1
    vote_fold[:, cv-1]=countvote.tolist()

    countvote=np.mean(vote[:,:],axis=1)
    vote_p[:, cv - 1] = countvote.tolist()

qual_all=[]
predict_p=np.mean(vote_fold[:,:],axis=1)
predicted_all=predict_p.copy()
predicted_all[predicted_all>=0.5]=1
predicted_all[predicted_all<0.5]=0
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
qual_all.append([accuracy, sens, spec, auc, auc_pr])
scio.savemat('./metrics/Test_MV.mat', {'qualified': qualified, 'predicted_all': predicted_all, 'predict_p': predict_p})

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

ROI =500
qualified=[]
for cv in [1,2,3,4,5]:
    time_step = 22
    vote = np.zeros([len(subInfo), int(ROI / 100)])
    for ROInum in range(100,ROI+100,100):
        FC_all = np.zeros((len(subInfo), time_step, ROInum, ROInum))
        subcount = 0
        for fn in subInfo['filename']:
            temp = scio.loadmat('/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/DFC/par' + str(ROInum) + '/' + fn)
            FC_all[subcount, :, :, :] = temp['DFC']
            subcount = subcount + 1
        FC_all = torch.from_numpy(FC_all)
        FC_all = torch.tensor(FC_all, dtype=torch.float32)

        dataset_test = TensorDataset(FC_all, y_data1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        model = MyModels.GraphConvolutionalLSTM_MIL0(ROInum=ROInum)
        model.load_state_dict(torch.load(
            './ckpt_obj_proj3/model'+str(neg)+'vs'+str(pos)+'_cv'+str(cv)+'_'+str(ROInum)+'_gcn-lstm_mil.pth'))
        model.cuda()
        predicted_all = []
        predict_p = []
        test_y_all = []
        model.eval()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(test_loader):
                test_y = test_y.view(-1)
                test_y = test_y.long()
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output, att = model(test_x)
                predict_p = predict_p + test_output[:, 1].tolist()
                predicted = torch.max(test_output.data, 1)[1]
                correct = (predicted == test_y).sum()
                accuracy = float(correct) / float(predicted.shape[0])
                test_y = test_y.cpu()
                predicted = predicted.cpu()
                predicted_all = predicted_all + predicted.tolist()
                test_y_all = test_y_all + test_y.tolist()

        vote[:, int((ROInum / 100)) - 1] = np.array(predicted_all)

    predict_p = np.mean(vote, axis=1)
    predicted_all=predict_p.copy()
    predicted_all[predicted_all>0.5]=1
    predicted_all[predicted_all<=0.5]=0
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


