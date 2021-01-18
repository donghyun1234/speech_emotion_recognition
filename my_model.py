import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import ipdb

import numpy as np
import pickle
import os
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dir(name):
    model_name = list()
    pred_name = list()

    for i in range(10):
        model_name.append(name + str(i) + '.pt')
        pred_name.append('./my_model/result/' + name + str(i) + '.pkl')

    return model_name, pred_name







### hidden과 cell을 넘겨주는 sliding LSTM
class LSTM_t(nn.Module):
    def __init__(self, feature_size, hidden_size1, bidirectional, n_layer=1, drop_out=0.3, n_out=1):
        super(LSTM_t, self).__init__()
        self.n_layer = n_layer  ### LSTM 몇 층을 사용할지 한층만 사용
        self.hidden_size2 = hidden_size1 * 2  ### 양방향 LSTM이므로 size가 두배로 증가

        ### 양 방향 LSTM 현재 모델에서 feature_size = 2560, hidden_size 256*2, n_layer = 1, 한 층일떄는 dropout의 비율은 의미없음(사용안함), bidirection
        self.LSTM = nn.LSTM(feature_size, hidden_size1, n_layer, dropout=drop_out, bidirectional=bidirectional)

    def forward(self, x, h, c, i):  ### x : input h : hidden c: cell_state, i는 initial h와 c를 넣기위해 사용하는것

        ### 첫번째 sliding LSTM일때 initial h와 c는 없음
        if i == 0:
            output, (hidden, cell1) = self.LSTM(x)
        ### 두번째 sliding LSTM부터 initial h와 c를 그 전의 sliding LSTM의 마지막 타임스텝의 것을 넣어줌
        else:
            output, (hidden, cell1) = self.LSTM(x, (h, c))

        a = torch.ones([output.size(0), 1, output.size(1)]).to(device) / output.size(1)  ### LSTM의 output을 평균내기위한 변수
        out_t_last = torch.matmul(a, output).squeeze()  ### 전체 output을 같은 비율로 해주기 위한 것
        return out_t_last, hidden, cell1


class model_2020(nn.Module):
    def __init__(self):
        super(model_2020, self).__init__()
        self.drop_p = 0.3  ### 드롭아웃 비율

        # dilated cnn에 들어가기전의 cnn
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), (1, 1), padding=1),  ### 채널 3 ->  128 kernel_size = 3, 3 stride 1, 1 padding = 1
            nn.LeakyReLU(),  ### 활성화 함수
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),  ### 채널 3 ->  128 kernel_size = 3, 3 stride 1, 1 padding = 1
            nn.LeakyReLU()  ### 활성화 함수
        )
        ### dilated cnn 을 사용하는 층 채널은 다음과 같이 128에서 그대로 128
        cnn_in = 128
        cnn_out = 64

        self.UFLB = nn.Sequential(nn.Conv2d(in_channels=cnn_in,  ### UFLB에서 사용하는 dilated cnn모두  채널 128 -> 128
                                            out_channels=cnn_out,
                                            ### kernel_size=3,3 stride=1,1 padding=2(input하고 output size를 똑같이 하려고)
                                            kernel_size=3,
                                            stride=(1, 1),
                                            padding=2,
                                            dilation=(2, 2)),
                                  nn.BatchNorm2d(cnn_out),  ### dilated cnn에서 나온결과를 정규화 해줌
                                  nn.LeakyReLU(),  ### 활성화 함수 비율을 안쓰면 0.01

                                  ### 위와 같은 구조가 총 3번 반복
                                  nn.Conv2d(in_channels=cnn_out,
                                            out_channels=cnn_out,
                                            kernel_size=3,
                                            stride=(1, 1),
                                            padding=2,
                                            dilation=(2, 2)),
                                  nn.BatchNorm2d(cnn_out),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(in_channels=cnn_out,
                                            out_channels=cnn_out,
                                            kernel_size=3,
                                            stride=(1, 1),
                                            padding=2,  # ?
                                            dilation=(2, 2)),
                                  nn.BatchNorm2d(cnn_out),
                                  nn.LeakyReLU())

        ### skip_connection도 위의 UFLB에서 사용한 구조와 똑같음
        self.skip_connection = nn.Sequential(nn.Conv2d(in_channels=cnn_in,
                                                       out_channels=cnn_out,
                                                       kernel_size=3,
                                                       stride=(1, 1),
                                                       padding=2,
                                                       dilation=(2, 2)),
                                             nn.BatchNorm2d(cnn_out),
                                             nn.LeakyReLU())

        ### LSTM의 모든 타임 스텝의 output값을 평균해주어 output을 내주는 sliding LSTM hidden과 cell을 넘겨줌
        self.LSTM = LSTM_t(1280, 256, bidirectional=True)

        ### 각 sliding LSTM의 attention의 가중치
        self.w_t0 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t1 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t2 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t3 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t4 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t5 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t6 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t7 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t8 = nn.Linear(512, 512, bias=False).cuda()
        self.w_t9 = nn.Linear(512, 512, bias=False).cuda()

        ### 각 sliding LSTM의 attention을 통과하고 attention 값을 구하고 난 후의 감정의 수로 차원을 축소 시킬 층
        self.out0 = nn.Linear(512, 4)
        self.out1 = nn.Linear(512, 4)
        self.out2 = nn.Linear(512, 4)
        self.out3 = nn.Linear(512, 4)
        self.out4 = nn.Linear(512, 4)
        self.out5 = nn.Linear(512, 4)
        self.out6 = nn.Linear(512, 4)
        self.out7 = nn.Linear(512, 4)
        self.out8 = nn.Linear(512, 4)
        self.out9 = nn.Linear(512, 4)

    def forward(self, x):

        ### dilated cnn 거치기 전의 cnn
        x = self.cnn(x)
        x = F.dropout(x, self.drop_p)


        ### 시간은 살리고 특징은 강화하기 위한 max_pooling
        x = F.max_pool2d(x, (1, 2))

        x1 = self.UFLB(x)  ### (dilated cnn + batch_norm + leakyrelu) *3
        x2 = self.skip_connection(x)  ### (dilated cnn + batch_norm + leakyrelu) *1
        x = x1 + x2


        x = x.reshape(x.size(0), x.size(2), -1)  ### LSTM에 넣기 위해 데이터를 3차원에서 2차원으로


        a = 30  ### 30프레임씩 데이터를 넣어줌
        n = int(x.size(1) / a)  ### 총 sliding LSTM의 갯수
        ### initial h, c를 넣기위해 변수를 선언하기 위해 해줌
        h = 1
        c = 1


        ### LSTM의 output을 넣어주기 위해 만든 변수
        output = torch.zeros(x.size(0), n, 512).to(device)

        ### sliding LSTM
        for i in range(0, n):
            out, h, c = self.LSTM(x[:, a * i: a * (i + 1)], h, c, i)
            output[:, i] = out

        time_attn_vector0 = F.softmax(
            torch.bmm(output[:, 0, :].unsqueeze(1), self.w_t0(output[:, 0, :].unsqueeze(1)).transpose(1, 2)), 2)

        out_t_last0 = torch.bmm(time_attn_vector0, output[:, 0, :].unsqueeze(1)).squeeze()

        out0 = F.softmax(self.out0(out_t_last0), dim=1)



        time_attn_vector1 = F.softmax(
            torch.bmm(output[:, 1, :].unsqueeze(1), self.w_t1(output[:, 0:2, :]).transpose(1, 2)), 2)


        out_t_last1 = torch.bmm(time_attn_vector1, output[:, 0:2, :]).squeeze()
        out1 = F.softmax(self.out1(out_t_last1), dim=1)

        time_attn_vector2 = F.softmax(
            torch.bmm(output[:, 2, :].unsqueeze(1), self.w_t2(output[:, 0:3, :]).transpose(1, 2)), 2)
        out_t_last2 = torch.bmm(time_attn_vector2, output[:, 0:3, :]).squeeze()
        out2 = F.softmax(self.out2(out_t_last2), dim=1)

        time_attn_vector3 = F.softmax(
            torch.bmm(output[:, 3, :].unsqueeze(1), self.w_t3(output[:, 0:4, :]).transpose(1, 2)), 2)
        out_t_last3 = torch.bmm(time_attn_vector3, output[:, 0:4, :]).squeeze()
        out3 = F.softmax(self.out3(out_t_last3), dim=1)

        time_attn_vector4 = F.softmax(
            torch.bmm(output[:, 4, :].unsqueeze(1), self.w_t4(output[:, 0:5, :]).transpose(1, 2)), 2)
        out_t_last4 = torch.bmm(time_attn_vector4, output[:, 0:5, :]).squeeze()
        out4 = F.softmax(self.out4(out_t_last4), dim=1)

        time_attn_vector5 = F.softmax(
            torch.bmm(output[:, 5, :].unsqueeze(1), self.w_t5(output[:, 0:6, :]).transpose(1, 2)), 2)
        out_t_last5 = torch.bmm(time_attn_vector5, output[:, 0:6, :]).squeeze()
        out5 = F.softmax(self.out5(out_t_last5), dim=1)

        time_attn_vector6 = F.softmax(
            torch.bmm(output[:, 6, :].unsqueeze(1), self.w_t6(output[:, 0:7, :]).transpose(1, 2)), 2)
        out_t_last6 = torch.bmm(time_attn_vector6, output[:, 0:7, :]).squeeze()
        out6 = F.softmax(self.out6(out_t_last6), dim=1)

        time_attn_vector7 = F.softmax(
            torch.bmm(output[:, 7, :].unsqueeze(1), self.w_t7(output[:, 0:8, :]).transpose(1, 2)), 2)
        out_t_last7 = torch.bmm(time_attn_vector7, output[:, 0:8, :]).squeeze()
        out7 = F.softmax(self.out7(out_t_last7), dim=1)

        time_attn_vector8 = F.softmax(
            torch.bmm(output[:, 7, :].unsqueeze(1), self.w_t7(output[:, 0:9, :]).transpose(1, 2)), 2)
        out_t_last8 = torch.bmm(time_attn_vector8, output[:, 0:9, :]).squeeze()
        out8 = F.softmax(self.out8(out_t_last8), dim=1)

        time_attn_vector9 = F.softmax(torch.bmm(output[:, 9, :].unsqueeze(1), self.w_t8(output).transpose(1, 2)), 2)
        out_t_last9 = torch.bmm(time_attn_vector9, output).squeeze()
        out9 = F.softmax(self.out9(out_t_last9), dim=1)

        ### 감정의 분포를 마지막 프레임을 제일 많이 보기위해 다음과 같이 계산
        out_sof = (out0 + out1) / 2
        out_sof = (out_sof + out2) / 2
        out_sof = (out_sof + out3) / 2
        out_sof = (out_sof + out4) / 2
        out_sof = (out_sof + out5) / 2
        out_sof = (out_sof + out6) / 2
        out_sof = (out_sof + out7) / 2
        out_sof = (out_sof + out8) / 2
        out_sof = (out_sof + out9) / 2

        ### 감정의 분포를 동등하게 보는 것
        # out_sof = (out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7 +out8+out9)/10
        return out_sof



def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = pickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = pickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def train(x, y, model, criterion, optimizer, device):
    model.train()

    x = torch.tensor(x, dtype=torch.float).to(device).permute(0,3,1,2)
    y = torch.tensor(y, dtype=torch.long).to(device).squeeze()

    y_pred = model(x)
    y_hot = F.one_hot(y,num_classes=4)
    loss = criterion[0](y_pred, y) + 0.01*criterion[1](y_pred, y_hot.type(torch.float))

    optimizer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()

    train_acc = torch.eq(y_pred.max(-1)[1], y).sum().item()

    return loss.item(), y_pred.cpu().detach().numpy(), train_acc


def eval(x, y, model, criterion, optimizer, device):
    model.eval()

    x = torch.tensor(x, dtype=torch.float).to(device).permute(0,3,1,2)
    y = torch.tensor(y, dtype=torch.long).to(device).squeeze()

    y_pred = model(x)
    y_hot = F.one_hot(y, num_classes=4)
    loss = criterion[0](y_pred, y) + 0.01*criterion[1](y_pred, y_hot.type(torch.float))

    return loss.item(), y_pred.cpu().detach().numpy()


def main(data_path, model_name, pred_name):
    num_epoch =  4000
    num_classes = 4
    batch_size = 30
    learning_rate = 0.00001
    # dropout_rate = 0.2

    checkpoint = './my_model/checkpoint/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####load data##########
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data(data_path)
    pernums_test = np.array(pernums_test)
    pernums_valid = np.array(pernums_valid)
    # train_label = dense_to_one_hot(train_label, num_classes)
    # valid_label = dense_to_one_hot(valid_label, num_classes)
    # Valid_label = dense_to_one_hot(Valid_label, num_classes)
    valid_size = valid_data.shape[0]
    dataset_size = train_data.shape[0]
    vnum = pernums_valid.shape[0]
    best_valid_uw = 0

    ##########define model###########
    model = model_2020().to(device)

    ##########tarin model###########
    criterion = [nn.CrossEntropyLoss().to(device), nn.MSELoss().to(device)]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.5, 0.999),  weight_decay = 1e-5)


    for i in range(num_epoch):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        tcost, tpred, tracc = train(train_data[start:end,:,:,:], train_label[start:end,:], model, criterion, optimizer, device)

        if i % 5 == 0:
            # for valid data
            valid_iter = divmod((valid_size), batch_size)[0]
            # Get predict result for confusion matrix
            y_pred_valid = np.empty((valid_size, num_classes), dtype=np.float32)
            y_valid = np.empty((vnum, 4), dtype=np.float32)
            index = 0
            cost_valid = 0
            if (valid_size < batch_size):
                loss, y_pred_valid = eval(valid_data, Valid_label, model, criterion, optimizer, device)
                cost_valid = cost_valid + np.sum(loss)
            for v in range(valid_iter):
                v_begin = v * batch_size
                v_end = (v + 1) * batch_size
                if (v == valid_iter - 1):
                    if (v_end < valid_size):
                        v_end = valid_size
                loss, y_pred_valid[v_begin:v_end, :] = eval(valid_data[v_begin:v_end], Valid_label[v_begin:v_end], model, criterion, optimizer, device)
                cost_valid = cost_valid + np.sum(loss)
            cost_valid = cost_valid / valid_size
            # calculate for segment valid data
            for s in range(vnum):
                y_valid[s, :] = np.max(y_pred_valid[index:index + pernums_valid[s], :], 0)
                index = index + pernums_valid[s]

            valid_acc_uw = recall(np.argmax(dense_to_one_hot(valid_label, num_classes), 1), np.argmax(y_valid, 1), average='macro')
            valid_conf = confusion(np.argmax(dense_to_one_hot(valid_label, num_classes), 1), np.argmax(y_valid, 1))
            if valid_acc_uw > best_valid_uw:
                best_valid_uw = valid_acc_uw
                best_valid_conf = valid_conf
                torch.save(model.state_dict(), os.path.join(checkpoint, model_name))

            # recall : The class is correctly recognised
            # Precision : An predict labeled as positive is indeed positive(confidence)
            # F1 Score : measure one label performance of classifier
            # UAR : mean(Recall) UAR is useful and can help to detect that one or more classifier are not good
            # but it does not give us any information about FP
            # That's why we should always have a look at the confusion matrix

            print("*****************************************************************")
            print("Epoch: %05d" % (i + 1))
            print("Training cost: %2.3g" % tcost)
            print("Training accuracy: %3.4g" % (tracc/batch_size))
            print("Valid cost: %2.3g" % cost_valid)
            print("Valid_UA: %3.4g" % valid_acc_uw)
            print("Best valid_UA: %3.4g" % best_valid_uw)
            print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print(valid_conf)
            print('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print(best_valid_conf)
            print("*****************************************************************")


def evaluate(data_path, model_name, pred_name):
    num_epoch = 2000
    num_classes = 4
    batch_size = 30
    learning_rate = 0.00001
    dropout_rate = 0.2

    checkpoint = './my_model/checkpoint/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####load data##########
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data(data_path)
    pernums_test = np.array(pernums_test)
    pernums_valid = np.array(pernums_valid)
    # train_label = dense_to_one_hot(train_label, num_classes)
    # valid_label = dense_to_one_hot(valid_label, num_classes)
    # Valid_label = dense_to_one_hot(Valid_label, num_classes)
    # test_label = dense_to_one_hot(test_label, num_classes)
    # Test_label = dense_to_one_hot(Test_label, num_classes)

    dataset_size = train_data.shape[0]
    valid_size = valid_data.shape[0]
    test_size = test_data.shape[0]
    vnum = pernums_valid.shape[0]
    tnum = pernums_test.shape[0]
    pred_test_uw = np.empty((tnum,4),dtype=np.float32)
    pred_test_w = np.empty((tnum,4),dtype=np.float32)
    best_valid_uw = 0
    best_valid_w = 0
    valid_iter = divmod((valid_size), batch_size)[0]
    test_iter = divmod((test_size), batch_size)[0]
    y_pred_valid = np.empty((valid_size, num_classes), dtype=np.float32)
    y_pred_test = np.empty((test_size, num_classes), dtype=np.float32)
    y_test = np.empty((tnum, 4), dtype=np.float32)
    y_valid = np.empty((vnum, 4), dtype=np.float32)

    ##########define model###########
    model = model_2020().to(device)

    ##########tarin model###########
    criterion = [nn.CrossEntropyLoss().to(device), nn.MSELoss().to(device)]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.5, 0.999),  weight_decay = 1e-5)

    ##########load model###########
    model.load_state_dict(torch.load(os.path.join(checkpoint, model_name)))
    model.eval()

    # for Validation
    index = 0
    cost_valid = 0

    if (valid_size < batch_size):
        loss, y_pred_valid = eval(valid_data, Valid_label, model, criterion, optimizer, device)
        cost_valid = cost_valid + np.sum(loss)
    for v in range(valid_iter):
        v_begin = v * batch_size
        v_end = (v + 1) * batch_size
        if (v == valid_iter - 1):
            if (v_end < valid_size):
                v_end = valid_size

        loss, y_pred_valid[v_begin:v_end, :] = eval(valid_data[v_begin:v_end], Valid_label[v_begin:v_end], model,
                                                    criterion, optimizer, device)
    cost_valid = cost_valid / valid_size
    # calculate for segment valid data
    for s in range(vnum):
        y_valid[s, :] = np.max(y_pred_valid[index:index + pernums_valid[s], :], 0)
        index = index + pernums_valid[s]

    valid_acc_uw = recall(np.argmax(dense_to_one_hot(valid_label,num_classes), 1), np.argmax(y_valid, 1), average='macro')
    valid_acc_w = recall(np.argmax(dense_to_one_hot(valid_label, num_classes),1), np.argmax(y_valid,1), average='weighted')
    valid_conf = confusion(np.argmax(dense_to_one_hot(valid_label, num_classes), 1), np.argmax(y_valid, 1))

    index = 0
    for t in range(test_iter):
        t_begin = t * batch_size
        t_end = (t + 1) * batch_size
        if (t == test_iter - 1):
            if (t_end < test_size):
                t_end = test_size
        loss, y_pred_test[t_begin:t_end, :] = eval(test_data[t_begin:t_end], Test_label[t_begin:t_end], model,
                                                    criterion, optimizer, device)
    # calculate for segment valid data
    for s in range(tnum):
        y_test[s, :] = np.max(y_pred_test[index:index + pernums_test[s], :], 0)
        index = index + pernums_test[s]

    # recall : The class is correctly recognised
    # Precision : An predict labeled as positive is indeed positive(confidence)
    # F1 Score : measure one label performance of classifier
    # UAR : mean(Recall) UAR is useful and can help to detect that one or more classifier are not good
    # but it does not give us any information about FP
    # That's why we should always have a look at the confusion matrix

    if valid_acc_uw > best_valid_uw:
        best_valid_uw = valid_acc_uw
        pred_test_uw = y_test
        test_acc_uw = recall(np.argmax(dense_to_one_hot(test_label, num_classes), 1), np.argmax(y_test, 1), average='macro')
        test_conf = confusion(np.argmax(dense_to_one_hot(test_label, num_classes), 1), np.argmax(y_test, 1))
        confusion_uw = test_conf
        flag = True

    if valid_acc_w > best_valid_w:
        best_valid_w = valid_acc_w
        pred_test_w = y_test
        test_acc_w = recall(np.argmax(dense_to_one_hot(test_label, num_classes), 1), np.argmax(y_test, 1), average='weighted')
        test_conf = confusion(np.argmax(dense_to_one_hot(test_label, num_classes), 1), np.argmax(y_test, 1))
        confusion_w = test_conf
        flag = True

    # export
    print("*****************************************************************")
    print("Valid cost: %2.3g" % cost_valid)
    print("Valid_UA: %3.4g" % valid_acc_uw)
    print("Valid_WA: %3.4g" % valid_acc_w)
    print("Best valid_UA: %3.4g" % best_valid_uw)
    print("Best valid_WA: %3.4g" % best_valid_w)
    print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
    print(valid_conf)
    print("Test_UA: %3.4g" % test_acc_uw)
    print("Test_WA: %3.4g" % test_acc_w)
    print('Test Confusion Matrix:["ang","sad","hap","neu"]')
    print(confusion_uw)
    print("*****************************************************************")
    if (flag):
        f = open(pred_name, 'wb')
        pickle.dump((best_valid_uw, best_valid_w, pred_test_w, test_acc_w, confusion_w, pred_test_uw,
                     test_acc_uw, confusion_uw,), f)
        f.close()
        flag = False


if __name__=='__main__':

    data_path = ['../data/ser_data/10fold/IEMOCAP_5M.pkl','../data/ser_data/10fold/IEMOCAP_5F.pkl','../data/ser_data/10fold/IEMOCAP_4M.pkl',
                 '../data/ser_data/10fold/IEMOCAP_4F.pkl','../data/ser_data/10fold/IEMOCAP_3M.pkl','../data/ser_data/10fold/IEMOCAP_3F.pkl',
                 '../data/ser_data/10fold/IEMOCAP_2M.pkl','../data/ser_data/10fold/IEMOCAP_2F.pkl','../data/ser_data/10fold/IEMOCAP_1M.pkl',
                 '../data/ser_data/10fold/IEMOCAP_1F.pkl',]

    # model_name = ['model1(5M).pt','model1(5F).pt','model1(4M).pt','model1(4F).pt','model1(3M).pt','model1(3F).pt','model1(2M).pt','model1(2F).pt','model1(1M).pt','model1(1F).pt']
    # pred_name = ['./result/model1(5M).pkl','./result/model1(5F).pkl','./result/model1(4M).pkl','./result/model1(4F).pkl','./result/model1(3M).pkl','./result/model1(3F).pkl','./result/model1(2M).pkl','./result/model1(2F).pkl','./result/model1(1M).pkl','./result/model1(1F).pkl']

    model_name, pred_name = make_dir('param1')

    for i in range(len(data_path)):
        main(data_path[i], model_name[i], pred_name[i])
        torch.cuda.empty_cache()
        evaluate(data_path[i], model_name[i], pred_name[i])
        torch.cuda.empty_cache()
