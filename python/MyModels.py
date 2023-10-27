import torch.nn.functional as F
import torch.nn as nn
import scipy.io as scio
import GNET
import torch
from collections import OrderedDict

class GCN_base(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GCN_base, self).__init__()

        #self.mgunets = nn.ModuleList()
        #for t in range(tsize):
            #self.mgunets.append(GUNET.MultiresolutionGUnet(nn.ReLU(),0.3))
        self.gcn = GUNET.GCN(ROInum, 1, nn.ReLU(),0.3)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(ROInum)
        self.fl1 = nn.Linear(ROInum,64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl2 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        batch_size = g.shape[0]
        ROInum = g.shape[2]

        fea = torch.zeros(g.size())
        for s in range(g.shape[0]):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g = g.cuda()
        out = torch.zeros(batch_size, ROInum)

        for s in range(batch_size):
            temp = self.gcn(g[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.softmax(out)

        return out

class GraphConvolutionalLSTM_base0(nn.Module):
    def __init__(self, ROInum, tsize, num_class=2):
        super(GraphConvolutionalLSTM_base0, self).__init__()

        self.gcn = GUNET.GCN(ROInum, 1, nn.ReLU(), 0.3)
        self.lstm = torch.nn.LSTM(ROInum, 100, 1, batch_first=True)

        self.bng = torch.nn.BatchNorm1d(tsize)
        self.fl1 = nn.Linear(tsize, 1)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.fl2 = nn.Linear(100, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.fl3 = nn.Linear(64, num_class)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        time_step = x.shape[1]
        ROInum = x.shape[2]

        fea = torch.zeros(x.size())
        for i in range(x.shape[0]):
            fea[i, :, :] = torch.eye(ROInum)
        fea = fea.cuda()
        x = x.cuda()

        out = torch.zeros(batch_size, time_step, ROInum)
        out = out.cuda()
        for i in range(time_step):
            for s in range(batch_size):
                temp = self.gcn(x[s, i, :, :], fea[s, i, :, :])
                temp.cuda()
                out[s, i, :] = torch.squeeze(temp)

        out.cuda()
        h0 = torch.zeros(1, batch_size, 100).cuda()
        c0 = torch.zeros(1, batch_size, 100).cuda()
        out, (h_n, h_c) = self.lstm(out.cuda(), (h0, c0))
        out.cuda()

        out = out.reshape([100 * batch_size, time_step])
        out = out.cuda()
        out = self.bng(out)
        # out = self.dropout(out)

        out = self.fl1(out)
        out = out.reshape([batch_size, 100])
        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl2(out)
        out = self.bn2.cuda()(out)
        out = F.relu(out)

        out = self.fl3(out)
        out = self.softmax(out)

        return out

class GraphConvolutionalLSTM_MIL0(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GraphConvolutionalLSTM_MIL0, self).__init__()
        self.L = 100
        self.D = 64
        self.K = 1

        self.gcn = GUNET.GCN(ROInum, 1, nn.ReLU(), 0.3)
        self.lstm = torch.nn.LSTM(ROInum, 100, 1, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.L * self.K, num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        time_step = x.shape[1]
        ROInum = x.shape[2]

        fea = torch.zeros(x.size())
        for i in range(x.shape[0]):
            fea[i, :, :] = torch.eye(ROInum)
        fea = fea.cuda()
        x = x.cuda()

        out = torch.zeros(batch_size, time_step, ROInum)
        out=out.cuda()
        for i in range(time_step):
            for s in range(batch_size):
                temp = self.gcn(x[s, i, :, :], fea[s, i, :, :])
                temp.cuda()
                out[s, i, :] = torch.squeeze(temp)

        out.cuda()
        h0 = torch.zeros(1, batch_size, 100).cuda()
        c0 = torch.zeros(1, batch_size, 100).cuda()
        out, (h_n, h_c) = self.lstm(out.cuda(), (h0, c0))
        out.cuda()

        A = self.attention(out)  # s*txK
        A = torch.transpose(A, 2, 1)  # s*Kxt
        Att = A
        A = F.softmax(A, dim=2)  # softmax over t

        M = torch.matmul(A, out)  # s*KxL
        M = M.reshape([batch_size, self.L * self.K])

        Y_prob = self.classifier(M)

        return Y_prob, Att

class GraphConvolutionalLSTM_MIL0_feature(nn.Module):
    def __init__(self,ROInum,num_class=2):
        super(GraphConvolutionalLSTM_MIL0_feature, self).__init__()
        self.L = 100
        self.D = 64
        self.K = 1

        self.gcn = GUNET.GCN(ROInum, 1, nn.ReLU(), 0.3)
        self.lstm = torch.nn.LSTM(ROInum, 100, 1, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.L * self.K, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        time_step = x.shape[1]
        ROInum = x.shape[2]

        fea = torch.zeros(x.size())
        for i in range(x.shape[0]):
            fea[i, :, :] = torch.eye(ROInum)
        fea = fea.cuda()
        x = x.cuda()

        out = torch.zeros(batch_size, time_step, ROInum)
        out=out.cuda()
        for i in range(time_step):
            for s in range(batch_size):
                temp = self.gcn(x[s, i, :, :], fea[s, i, :, :])
                temp.cuda()
                out[s, i, :] = torch.squeeze(temp)

        out.cuda()
        h0 = torch.zeros(1, batch_size, 100).cuda()
        c0 = torch.zeros(1, batch_size, 100).cuda()
        out, (h_n, h_c) = self.lstm(out.cuda(), (h0, c0))
        out.cuda()

        A = self.attention(out)  # s*txK
        A = torch.transpose(A, 2, 1)  # s*Kxt
        Att = A
        A = F.softmax(A, dim=2)  # softmax over t

        M = torch.matmul(A, out)  # s*KxL
        M = M.reshape([batch_size, self.L * self.K])

        Y_prob = self.classifier(M)

        return M, Att

class MGCUNET_LSTM_MIL_fusion0(nn.Module):
    def __init__(self,cv):
        super(MGCUNET_LSTM_MIL_fusion0, self).__init__()
        self.cv=cv
        self.gclstm1 = GraphConvolutionalLSTM_MIL0(ROInum=100)
        self.gclstm2 = GraphConvolutionalLSTM_MIL0(ROInum=200)
        self.gclstm3 = GraphConvolutionalLSTM_MIL0(ROInum=300)
        self.gclstm4 = GraphConvolutionalLSTM_MIL0(ROInum=400)
        self.gclstm5 = GraphConvolutionalLSTM_MIL0(ROInum=500)

        self.alpha = nn.Parameter(torch.FloatTensor(torch.randn([5])))

        #self.proj = nn.Linear(5,2)
        self.softmax = nn.Softmax()
    def init(self):
        self.gclstm1.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '100_gcn-lstm_mil.pth'))
        self.gclstm2.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '200_gcn-lstm_mil.pth'))
        self.gclstm3.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '300_gcn-lstm_mil.pth'))
        self.gclstm4.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '400_gcn-lstm_mil.pth'))
        self.gclstm5.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '500_gcn-lstm_mil.pth'))

    def forward(self, g1, g2, g3, g4, g5):
        Y1,att = self.gclstm1(g1)
        Y2,att = self.gclstm2(g2)
        Y3,att = self.gclstm3(g3)
        Y4,att = self.gclstm4(g4)
        Y5,att = self.gclstm5(g5)

        out = Y1.mul(self.alpha[0])+Y2.mul(self.alpha[1])+Y3.mul(self.alpha[2])+Y4.mul(self.alpha[3])+Y5.mul(self.alpha[4])
        out = self.softmax(out)

        return out

    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith('module'):
            state_idx = 1
        else:
            state_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ','.join(k.split('.')[state_idx:])
            new_state_dict[name] = v
        return new_state_dict

class MGCUNET_LSTM_MIL_fusion2(nn.Module):
    def __init__(self,cv):
        super(MGCUNET_LSTM_MIL_fusion2, self).__init__()

        self.cv=cv

        self.gclstm1 = GraphConvolutionalLSTM_MIL0_feature(ROInum=100)
        self.gclstm2 = GraphConvolutionalLSTM_MIL0_feature(ROInum=200)
        self.gclstm3 = GraphConvolutionalLSTM_MIL0_feature(ROInum=300)
        self.gclstm4 = GraphConvolutionalLSTM_MIL0_feature(ROInum=400)
        self.gclstm5 = GraphConvolutionalLSTM_MIL0_feature(ROInum=500)

        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.bn3 = torch.nn.BatchNorm1d(100)
        self.bn4 = torch.nn.BatchNorm1d(100)
        self.bn5 = torch.nn.BatchNorm1d(100)

        self.fl1 = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(500, 256),
            nn.ReLU()
        )

        self.fl2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Softmax()
        )

    def init(self):
        self.gclstm1.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '100_gcn-lstm_mil.pth'))
        self.gclstm2.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '200_gcn-lstm_mil.pth'))
        self.gclstm3.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '300_gcn-lstm_mil.pth'))
        self.gclstm4.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '400_gcn-lstm_mil.pth'))
        self.gclstm5.load_state_dict(
            torch.load('./model_DFC/model0vs1' + '_cv' + str(self.cv) + '_' + '500_gcn-lstm_mil.pth'))


    def forward(self, g1, g2, g3, g4, g5):
        M1, Att1 = self.gclstm1(g1)
        M1 = self.bn1(M1)
        M2, Att2 = self.gclstm2(g2)
        M2 = self.bn2(M2)
        M3, Att3 = self.gclstm3(g3)
        M3 = self.bn3(M3)
        M4, Att4 = self.gclstm4(g4)
        M4 = self.bn4(M4)
        M5, Att5 = self.gclstm5(g5)
        M5 = self.bn5(M5)

        M = torch.cat((M1, M2, M3, M4, M5), 1)
        #Att = torch.cat((Att1, Att2, Att3, Att4, Att5), 1)

        M = self.fl1(M)
        M = self.fl2(M)

        Y_prob = self.classifier(M)

        return Y_prob
