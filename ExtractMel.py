#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:32:28 2018

@author: hxj
"""

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import ipdb
#import base
#import sigproc
eps = 1e-5


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def getlogspec(signal,samplerate=16000,winlen=0.02,winstep=0.01,
               nfilt=26,nfft=399,lowfreq=0,highfreq=None,preemph=0.97,
               winfunc=lambda x:np.ones((x,))):
    highfreq= highfreq or samplerate/2
    signal = ps.sigproc.preemphasis(signal,preemph)
    frames = ps.sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames,nfft)
    return pspec


def read_file(filename, norm):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    #wavedata = np.fromstring(str_data, dtype = np.short)
    wavedata = np.frombuffer(str_data, dtype=np.short)
    if norm == True:
        wavedata = (wavedata*1.0/max(abs(wavedata))).astype(np.float32)  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def zscore(data,mean,std):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        data[i,:,:,0] = (data[i,:,:,0]-mean)/(std)
    return data


def normalization(data):
    '''
    #apply zscore
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    train_data = zscore(train_data,mean,std)
    test_data = zscore(test_data,mean,std)
    '''
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    data = (data-mean)/std
    return data


def mapminmax(data):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        min = np.min(data[i,:,:,0])
        max = np.max(data[i,:,:,0])
        data[i,:,:,0] = (data[i,:,:,0] - min)/((max - min)+eps)
    return data


def generate_label(emotion,classnum):
    label = -1
    if(emotion == 'ang'):
        label = 0
    elif(emotion == 'sad'):
        label = 1
    elif(emotion == 'hap'):
        label = 2
    elif(emotion == 'neu'):
        label = 3
    elif(emotion == 'fear'):
        label = 4
    else:
        label = 5
    return label


def load_data(index):
    f = open('../data/ser_data/zscore/zscore'+str(1)+'.pkl','rb')
    #f = open('../data/ser_data/zscore/zscore' + str(index) + '.pkl', 'rb')
    mean1,std1,mean2,std2,mean3,std3 = pickle.load(f)
    return mean1,std1,mean2,std2,mean3,std3
        
'''
Train data : 각 클래스별 300개씩 랜덤으로 취함 (1200,300,40,3)
train data : 전체 train 데이터
Test label : session5의 남자를 2 sec segment 한 데이터 
test label : session5의 남자를 300 timestep으로 segment 한 데이터
Valid label : session5의 여자를 2 sec segment 한 데이터
valid label : session5의 여자를 300 timestep으로 segment 한 데이터
'''


def read_IEMOCAP(dataset_name):
    eps = 1e-5
    tnum = 259 #the number of test utterance
    vnum = 298
    test_num = 420 #the number of test 2s segments
    valid_num = 436
    train_num = 2928
    filter_num = 40
    pernums_test = list()#remerber each utterance contain how many segments
    pernums_valid = list()
    rootdir = '../data/ser_data/IEMOCAP_full_release/'

    print(dataset_name)
    idx = [10,5,4,3,2,1]
    test_session = int(dataset_name[0])
    session_list = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    del session_list[test_session-1]

    mean1,std1,mean2,std2,mean3,std3 = load_data(idx[test_session])
    #2774
    hapnum = 434#2
    angnum = 433#0
    neunum = 1262#3
    sadnum = 799#1
    pernum = 300#np.min([hapnum,angnum,sadnum,neunum])
    #valid_num = divmod((train_num),10)[0]
#    train_label = np.empty((train_num,1), dtype = np.int8)
#    test_label = np.empty((tnum,1), dtype = np.int8)
#    valid_label = np.empty((vnum,1), dtype = np.int8)
#    Test_label = np.empty((test_num,1), dtype = np.int8)
#    Valid_label = np.empty((valid_num,1), dtype = np.int8)
#    train_data = np.empty((train_num,300,filter_num,3),dtype = np.float32)
#    test_data = np.empty((test_num,300,filter_num,3),dtype = np.float32)
#    valid_data = np.empty((valid_num,300,filter_num,3),dtype = np.float32)

    train_label = list()
    test_label = list()
    valid_label = list()
    Test_label = list()
    Valid_label = list()
    train_data1 = list()
    train_data2 = list()
    train_data3 = list()
    test_data1 = list()
    test_data2 = list()
    test_data3 = list()
    valid_data1 = list()
    valid_data2 = list()
    valid_data3 = list()

    tnum = 0
    vnum = 0
    train_num = 0
    test_num = 0
    valid_num = 0
    train_emt = {'hap':0,'ang':0,'neu':0,'sad':0 }
    test_emt = {'hap':0,'ang':0,'neu':0,'sad':0 }
    valid_emt = {'hap':0,'ang':0,'neu':0,'sad':0 }
    for speaker in os.listdir(rootdir):
        if(speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir,speaker,'sentences/wav')
            emoevl = os.path.join(rootdir,speaker,'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if(sess[7] == 'i'):
                    emotdir = emoevl+'/'+sess+'.txt'
                    #emotfile = open(emotdir)
                    emot_map = {}
                    with open(emotdir,'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if(line[0] == '['):
                                t = line.split()
                                emot_map[t[3]] = t[4]
                                
        
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        #wavname = filename[-23:-4]
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        if(emotion in ['hap','ang','neu','sad']):
                            data, time, rate = read_file(filename, True)
                            mel_spec = ps.logfbank(data,rate, nfilt = filter_num)
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)
                            #apply zscore

                            time = mel_spec.shape[0]
                            if(speaker in session_list):
                                #training set
                                if(time <= 300):
                                    part = mel_spec
                                    delta11 = delta1
                                    delta21 = delta2
                                    part = np.pad(part,((0,300 - part.shape[0]),(0,0)),'constant',constant_values = 0)
                                    delta11 = np.pad(delta11,((0,300 - delta11.shape[0]),(0,0)),'constant',constant_values = 0)
                                    delta21 = np.pad(delta21,((0,300 - delta21.shape[0]),(0,0)),'constant',constant_values = 0)
                                    train_data1.append((part -mean1)/(std1+eps))
                                    train_data2.append((delta11 - mean2)/(std2+eps))
                                    train_data3.append((delta21 - mean3)/(std3+eps))

                                    em = generate_label(emotion,6)
                                    train_label.append(em)
                                    train_emt[emotion] = train_emt[emotion] + 1
                                    train_num = train_num + 1
                                else:
                                    if(emotion in ['ang','neu','sad']):
                                         
                                        for i in range(2):
                                            if(i == 0):
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                begin = time - 300
                                                end = time
                                          
                                            part = mel_spec[begin:end,:]
                                            delta11 = delta1[begin:end,:]
                                            delta21 = delta2[begin:end,:]
                                            train_data1.append((part -mean1)/(std1+eps))
                                            train_data2.append((delta11 - mean2)/(std2+eps))
                                            train_data3.append((delta21 - mean3)/(std3+eps))

                                            em = generate_label(emotion,6)
                                            train_label.append(em)
                                            train_emt[emotion] = train_emt[emotion] + 1
                                            train_num = train_num + 1
                                    else:
                                        frames = divmod(time-300,100)[0] + 1
                                        for i in range(frames):
                                            begin = 100*i
                                            end = begin + 300
                                            part = mel_spec[begin:end,:]
                                            delta11 = delta1[begin:end,:]
                                            delta21 = delta2[begin:end,:]
                                            train_data1.append((part -mean1)/(std1+eps))
                                            train_data2.append((delta11 - mean2)/(std2+eps))
                                            train_data3.append((delta21 - mean3)/(std3+eps))
                                            em = generate_label(emotion,6)
                                            train_label.append(em)
                                            train_emt[emotion] = train_emt[emotion] + 1
                                            train_num = train_num + 1
                            else:
                                em = generate_label(emotion,6)
                                #if(wavname[-4] == 'M'):
                                if(wavname[-4] == dataset_name[1]):
                                    #test_set
                                    test_label.append(em)
                                    if(time <= 300):
                                        pernums_test.append(1)
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2
                                        part = np.pad(part,((0,300 - part.shape[0]),(0,0)),'constant',constant_values = 0)
                                        delta11 = np.pad(delta11,((0,300 - delta11.shape[0]),(0,0)),'constant',constant_values = 0)
                                        delta21 = np.pad(delta21,((0,300 - delta21.shape[0]),(0,0)),'constant',constant_values = 0)
                                        test_data1.append((part -mean1)/(std1+eps))
                                        test_data2.append((delta11 - mean2)/(std2+eps))
                                        test_data3.append((delta21 - mean3)/(std3+eps))
                                        test_emt[emotion] = test_emt[emotion] + 1
                                        Test_label.append(em)
                                        test_num = test_num + 1
                                        tnum = tnum + 1
                                    else:
                                        pernums_test.append(2)
                                        tnum = tnum + 1
                                        for i in range(2):
                                            if(i == 0):
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                end = time
                                                begin = time - 300
                                            part = mel_spec[begin:end,:]
                                            delta11 = delta1[begin:end,:]
                                            delta21 = delta2[begin:end,:]
                                            test_data1.append((part -mean1)/(std1+eps))
                                            test_data2.append((delta11 - mean2)/(std2+eps))
                                            test_data3.append((delta21 - mean3)/(std3+eps))

                                            test_emt[emotion] = test_emt[emotion] + 1
                                            Test_label.append(em)
                                            test_num = test_num + 1

                                else:
                                    #valid_set
                                    em = generate_label(emotion,6)
                                    valid_label.append(em)
                                    if(time <= 300):
                                        pernums_valid.append(1)
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2
                                        part = np.pad(part,((0,300 - part.shape[0]),(0,0)),'constant',constant_values = 0)
                                        delta11 = np.pad(delta11,((0,300 - delta11.shape[0]),(0,0)),'constant',constant_values = 0)
                                        delta21 = np.pad(delta21,((0,300 - delta21.shape[0]),(0,0)),'constant',constant_values = 0)
                                        valid_data1.append((part -mean1)/(std1+eps))
                                        valid_data2.append((delta11 - mean2)/(std2+eps))
                                        valid_data3.append((delta21 - mean3)/(std3+eps))
                                        valid_emt[emotion] = valid_emt[emotion] + 1
                                        Valid_label.append(em)
                                        valid_num = valid_num + 1
                                        vnum = vnum + 1
                                    else:
                                        pernums_valid.append(2)
                                        vnum = vnum + 1
                                        for i in range(2):
                                            if(i == 0):
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                end = time
                                                begin = time - 300
                                            part = mel_spec[begin:end,:]
                                            delta11 = delta1[begin:end,:]
                                            delta21 = delta2[begin:end,:]
                                            valid_data1.append((part -mean1)/(std1+eps))
                                            valid_data2.append((delta11 - mean2)/(std2+eps))
                                            valid_data3.append((delta21 - mean3)/(std3+eps))
                                            valid_emt[emotion] = valid_emt[emotion] + 1
                                            Valid_label.append(em)
                                            valid_num = valid_num + 1
                                     
                                    
                                 
                        else:
                            pass
    
    
    
    hap_index = list()
    neu_index = list()
    sad_index = list()
    ang_index = list()
    h2 = 0
    a0 = 0
    n3 = 0
    s1 = 0
#    ipdb.set_trace()
    for l in range(train_num):
        if(train_label[l] == 0):
            ang_index.append(l)
            a0 = a0 + 1
        elif (train_label[l] == 1):
            sad_index.append(l)
            s1 = s1 + 1
        elif (train_label[l] == 2):
            hap_index.append(l)
            h2 = h2 + 1
        else:
            neu_index.append(l)
            n3 = n3 + 1

    for m in range(1):
        # np.random.shuffle(neu_index)
        # np.random.shuffle(hap_index)
        # np.random.shuffle(sad_index)
        # np.random.shuffle(ang_index)
        #define emotional array
        hap_label = np.empty((pernum,1), dtype = np.int8)
        ang_label = np.empty((pernum,1), dtype = np.int8)
        sad_label = np.empty((pernum,1), dtype = np.int8)
        neu_label = np.empty((pernum,1), dtype = np.int8)
        hap_data = np.empty((pernum,300,filter_num,3),dtype = np.float32)
        neu_data = np.empty((pernum,300,filter_num,3),dtype = np.float32)
        sad_data = np.empty((pernum,300,filter_num,3),dtype = np.float32)
        ang_data = np.empty((pernum,300,filter_num,3),dtype = np.float32)

        train_data1 = np.reshape(train_data1, (-1,300, 40))
        train_data2 = np.reshape(train_data2, (-1,300, 40))
        train_data3 = np.reshape(train_data3, (-1,300, 40))
        test_data1 = np.reshape(test_data1, (-1,300, 40))
        test_data2 = np.reshape(test_data2, (-1,300, 40))
        test_data3 = np.reshape(test_data3, (-1,300, 40))
        valid_data1 = np.reshape(valid_data1, (-1,300, 40))
        valid_data2 = np.reshape(valid_data2, (-1,300, 40))
        valid_data3 = np.reshape(valid_data3, (-1,300, 40))

        train_list = [train_data1, train_data2, train_data3]
        test_list = [test_data1, test_data2, test_data3]
        valid_list = [valid_data1, valid_data2, valid_data3]

        train_data = np.stack(train_list, axis=3)
        test_data = np.stack(test_list, axis=3)
        valid_data = np.stack(valid_list, axis=3)
        train_label = np.reshape(np.array(train_label),(-1,1))
        test_label = np.reshape(np.array(test_label),(-1,1))
        valid_label = np.reshape(np.array(valid_label),(-1,1))
        Test_label = np.reshape(np.array(Test_label),(-1,1))
        Valid_label = np.reshape(np.array(Valid_label),(-1,1))


        hap_data = train_data[hap_index[0:pernum]].copy()
        hap_label = train_label[hap_index[0:pernum]].copy()
        ang_data = train_data[ang_index[0:pernum]].copy()
        ang_label = train_label[ang_index[0:pernum]].copy()
        sad_data = train_data[sad_index[0:pernum]].copy()
        sad_label = train_label[sad_index[0:pernum]].copy()
        neu_data = train_data[neu_index[0:pernum]].copy()
        neu_label = train_label[neu_index[0:pernum]].copy()
        train_num = 4*pernum
    
        Train_label = np.empty(train_label.shape, dtype = np.int8)
        Train_data = np.empty(train_data.shape,dtype = np.float32)
        Train_data[0:pernum] = hap_data
        Train_label[0:pernum] = hap_label
        Train_data[pernum:2*pernum] = sad_data
        Train_label[pernum:2*pernum] = sad_label  
        Train_data[2*pernum:3*pernum] = neu_data
        Train_label[2*pernum:3*pernum] = neu_label 
        Train_data[3*pernum:4*pernum] = ang_data
        Train_label[3*pernum:4*pernum] = ang_label
    
        arr = np.arange(train_num)
        # np.random.shuffle(arr)
        Train_data = Train_data[arr[0:]]
        Train_label = Train_label[arr[0:]]
        print('train_data',train_data.shape)
        print('Train_data',Train_data.shape)
        print('valid_data',valid_data.shape)
        print('Valid_label',Valid_label.shape)
        print('valid_label',valid_label.shape)
        print('test_data',test_data.shape)
        print('Test_label',Test_label.shape)
        print('test_label',test_label.shape)
        print('train_emt',train_emt)
        print('test_emt',test_emt)
        print('valid_emt',valid_emt)
        #print test_label[0:500,:]
        #f=open('./CASIA_40_delta.pkl','wb')
        #output = './IEMOCAP40.pkl'
        output = '../data/ser_data/10fold/IEMOCAP_'+dataset_name+'.pkl'
        pernums_test = np.array(pernums_test)
        pernums_valid = np.array(pernums_valid)

        f=open(output,'wb')
        pickle.dump((Train_data,Train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid),f)
        f.close()
    return


if __name__=='__main__':
    # make as 10fold dataset(ex.5M,5F)
    dataset_name = ['5M','5F','4M','4F','3M','3F','2M','2F','1M','1F']
    for i in range(len(dataset_name)):
        read_IEMOCAP(dataset_name[i])

    #print "test_num:", test_num
    #print "train_num:", train_num
#    n = wgn(x, 6)
#    xn = x+n # 增加了6dBz信噪比噪声的信号
