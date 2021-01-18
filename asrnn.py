import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ipdb

import numpy as np
import pickle
import os
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion


def make_dir(name):
    model_name = list()
    pred_name = list()

    for i in range(10):
        model_name.append(name + str(i) + '.pt')
        pred_name.append('./result/' + name + str(i) + '.pkl')

    return model_name, pred_name


class TraditionalCNN(nn.Module):
    def __init__(self, in_chennel=3, out_chennel=128, conv_kernel=(3,3), conv_stride=(1,1),conv_padding=(1,1), pool_kernel=(2,2), pool_stride=(2,2)):
        super(TraditionalCNN,self).__init__()

        # input_size (batch, channel, feature, timestep)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_chennel, out_chennel, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.MaxPool2d(pool_kernel, stride=pool_stride),
            nn.BatchNorm2d(out_chennel,momentum=0.1,affine=False, track_running_stats=True),
            nn.ReLU()
        )

        # self.cnn.apply(self.init_he)

    def forward(self, x):
        return self.cnn(x)

    def init_he(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(m.bias, 0.1)
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(m.bias, 0.1)


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()

        self.W_omega = nn.Parameter(torch.FloatTensor(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.FloatTensor(attention_size))
        self.u_omega = nn.Parameter(torch.FloatTensor(attention_size))

        nn.init.kaiming_normal_(self.W_omega, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.b_omega,std=0.01)
        nn.init.normal_(self.u_omega,std=0.01)

    def forward(self, x, time_major):

        if time_major:
            x = x.permute(1,0,2)

        v = F.relu(torch.tensordot(x, self.W_omega, dims=1) + self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alpha = F.softmax(vu, dim=1)

        output = (x * alpha.unsqueeze(-1)).mean(1)

        return output, alpha

class asrnn(nn.Module):
    def __init__(self):
        super(asrnn, self).__init__()

        self.cnn1 = TraditionalCNN(in_chennel=3, out_chennel=64, conv_kernel=(3,3), conv_stride=(1,1),conv_padding=(1,1), pool_kernel=(2,2), pool_stride=(2,2))
        self.cnn2 = TraditionalCNN(in_chennel=64, out_chennel=128, conv_kernel=(3,3), conv_stride=(1,1),conv_padding=(1,1), pool_kernel=(1,1), pool_stride=(1,1))
        self.cnn3 = TraditionalCNN(in_chennel=128, out_chennel=128, conv_kernel=(3,3), conv_stride=(1,1),conv_padding=(1,1), pool_kernel=(1,1), pool_stride=(1,1))

        self.linear1 = nn.Linear(in_features=20*128, out_features=768)
        self.linear2 = nn.Linear(in_features=512, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=4)

        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)

        self.attention = Attention(256, 1)

        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.linear3.weight, gain=1)

        self.batch1 = nn.BatchNorm1d(768)
        self.batch2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        input_size = x.size()
        x = x.view(-1,input_size[1]*input_size[2],input_size[3]).permute(0,2,1)
        x = x.reshape(-1,input_size[1]*input_size[2])
        x = self.linear1(x)
        x = F.leaky_relu(x,0.01)
        x = self.batch1(x)
        x = x.view(-1, input_size[3], 768)

        rnn_list, len = self.slice(x.permute(0,2,1), window=10, shift=5)
        slice_list = list()
        
        # RNN (time, batch, hidden)
        for i in range(len):
            slice_x = rnn_list[i].permute(2,0,1)
            slice_out, (bn, cn) = self.bilstm(slice_x)
            slice_list.append(slice_out[-1])

        rnn_out = torch.stack(slice_list, dim=0)
        rnn_out2,_ = self.lstm(rnn_out)
        attened, alpha = self.attention(rnn_out,True)
        output = torch.cat([attened, rnn_out2[-1]], dim=1)

        output = F.leaky_relu(self.linear2(output), 0.01)
        output = self.batch2(output)
        output = self.linear3(output)

        return output

    def slice(self, x, window, shift):
        output = list()
        idx = divmod((x.size()[2] - window), shift)[0] + 1
        for i in range(idx):
            if i == (idx -1):
                output.append(x[:,:, -window:])
            else:
                output.append(x[:, :, i:i+window])
        return output, idx

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

    x = torch.tensor(x, dtype=torch.float).to(device).permute(0,3,2,1)
    y = torch.tensor(y, dtype=torch.long).to(device).squeeze()

    y_pred = model(x)
    y_hot = F.one_hot(y,num_classes=4)
    #loss = criterion[0](y_pred, y) + 0.01*criterion[1](y_pred, y_hot.type(torch.float))
    loss = criterion[0](y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()

    train_acc = torch.eq(y_pred.max(-1)[1], y).sum().item()

    return loss.item(), y_pred.cpu().detach().numpy(), train_acc


def eval(x, y, model, criterion, optimizer, device):
    model.eval()

    x = torch.tensor(x, dtype=torch.float).to(device).permute(0,3,2,1)
    y = torch.tensor(y, dtype=torch.long).to(device).squeeze()

    y_pred = model(x)
    y_hot = F.one_hot(y, num_classes=4)
    # loss = criterion[0](y_pred, y) + 0.01*criterion[1](y_pred, y_hot.type(torch.float))
    loss = criterion[0](y_pred, y)

    return loss.item(), y_pred.cpu().detach().numpy()


def main(data_path, model_name, pred_name):
    num_epoch = 5000
    num_classes = 4
    batch_size = 30
    learning_rate = 0.00001
    dropout_rate = 0.2

    checkpoint = './checkpoint/'

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
    model = asrnn().to(device)

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
    num_epoch = 5000
    num_classes = 4
    batch_size = 60
    learning_rate = 0.00001
    dropout_rate = 0.2

    checkpoint = './checkpoint/'

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
    model = asrnn().to(device)

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

    data_path = ['./10fold/IEMOCAP5M.pkl','./10fold/IEMOCAP5F.pkl','./10fold/IEMOCAP4M.pkl','./10fold/IEMOCAP4F.pkl','./10fold/IEMOCAP3M.pkl','./10fold/IEMOCAP3F.pkl','./10fold/IEMOCAP2M.pkl','./10fold/IEMOCAP2F.pkl','./10fold/IEMOCAP1M.pkl','./10fold/IEMOCAP1F.pkl',]
    # model_name = ['model1(5M).pt','model1(5F).pt','model1(4M).pt','model1(4F).pt','model1(3M).pt','model1(3F).pt','model1(2M).pt','model1(2F).pt','model1(1M).pt','model1(1F).pt']
    # pred_name = ['./result/model1(5M).pkl','./result/model1(5F).pkl','./result/model1(4M).pkl','./result/model1(4F).pkl','./result/model1(3M).pkl','./result/model1(3F).pkl','./result/model1(2M).pkl','./result/model1(2F).pkl','./result/model1(1M).pkl','./result/model1(1F).pkl']

    model_name, pred_name = make_dir('param1')

    for i in range(len(data_path)):
        main(data_path[i], model_name[i], pred_name[i])
        torch.cuda.empty_cache()
        evaluate(data_path[i], model_name[i], pred_name[i])
        torch.cuda.empty_cache()
