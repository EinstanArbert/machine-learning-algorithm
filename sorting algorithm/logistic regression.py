#!usr/bin/env python3
#coding=utf-8

import numpy as np


def sig(x):
    return 1.0/(1+np.exp(-x))

def lr_train_bgd(feature,label,maxCycle,alpha):
    n=np.shape(feature)[1]
    w=np.mat(np.ones((n,1)))
    i=0
    while i<=maxCycle:
        i+=1
        h=sig(feature*w)
        err=label-h
        if i%100==0:
            print('\t---iter='+str(i)+\
                ",train error="+str(error_rate(h,label)))
            w+=alpha*feature.T*err
    return w

def error_rate(h,label):
    m=np.shape(h)[0]
    sum_err=0.0
    for i in xrange(m):
        if h[i,0]>0 and (1-h[i,0])>0:
            sum_err-=(label[i,0]*np.log(h[i,0])+\
                (1-label[i,0])*np.log(1-h[i,0]))
        else:
            sum_err-=0
    return sum_err/m

def load_data(file_name):
    f=open(file_name)
    feature_data=[]
    label_data=[]
    for line in f.readlines():
        feature_tmp=[]
        label_tmp=[]
        lines=line.strip().split("\t")
        feature_tmp.append(1)
        for i in xrange (len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))
        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data)

def save_model(file_name,w):
    m=np.shape(w)[0]
    f_w=open(file_name,"w")
    w_array=[]
    for i in xrange(m):
        w_array.append(str(w[i,0]))
    f_w.write("\t".join(w_array))
    f_w.close()

def load_weight(w):
    f=open(w)
    w=[]
    for line in f.readlines():
        lines=line.strip().split("\t")
        w_tmp=[]
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    return np.mat(w)

def load_data(file_name,n):
    f=open(file_name)
    feature_data=[]
    for line in f.readlines():
        feature_tmp=[]
        lines=line.strip().split("\t")
        if len(lines)!=n-1:
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)

def predeict(data,w):
    h=sig(data*w.T)
    m=np.shape(h)[0]
    for i in xrange(m):
        if h[i,0]<0.5:
            h[i,0]=0.0
        else:
            h[i,0]=1.0
    return h

def save_result(file_name,result):
    m=np.shape(result)[0]
    tmp=[]
    for i in xrange(m):
        tmp.append(str(m[i,0]))
    f_result=open(file_name,"w")
    f_result.write("\t".join(tmp))
    f_result.close()

if __name__ == "__main__":
    feature,label=load_data("data.txt")
    w=lr_train_bgd(feature,label,1000,0.01)
    save_model("weight",w)

    # predict
    # w=load_weight("weights")
    # n=np.shape(w)[1]
    # testData=load_data("test_data",n)
    # h=predeict(testData,w)
    # save_result("result",h)