import numpy as np  
import pandas as pd  
import random as rd  
import csv
from sklearn import preprocessing

class knn:
    def __init__(self, name):
        self.train_file = name
        self.feature=[]
        self.label=[]
    def train(self):
        self.feature,self.label=gen_model(self.train_file)       
    def test(self,x,k):
        result=knn_classify(self.feature,self.label,k,x)
        return result


def knn_classify(train_set,train_label,k,x):
    data_mat=(train_set)
    data_len = len(data_mat)
    diff_mat = (np.tile(x,(data_len,1))-data_mat)**2
    dist=diff_mat.sum(axis=1)
    sorted_idx = dist.argsort()
    label_count={}
    curr_max_vote= 0;curr_max_label=0
    for i in range(k):
        curr_label=label[sorted_idx[i]]
        label_count[curr_label]=label_count.get(curr_label,0)+1
        if(label_count[curr_label] > curr_max_vote):
            curr_max_vote = label_count[curr_label]
            curr_max_label = curr_label
    return curr_max_label

def gen_model(name):
    #df = pd.read_csv('datingTestSet.txt') 
    txt_name = name +".txt"
    csv_name = name + ".csv"  
    file = open(csv_name,'wb') 
    my_save=csv.writer(file) 
    fr = open(txt_name)
    line_mat = fr.readlines()
    data_mat=[]
    for line in line_mat:
        line=line.strip()
        line_list = line.split('\t')
        data_mat.append(line_list)
        my_save.writerow(line_list)
    file.close()
    feature,label=pre_data(csv_name)
    return feature,label


def pre_data(csv_name):
    df=pd.read_csv(csv_name,names=['f1','f2','f3','label'])
    df['label_id']=pd.factorize(df['label'])[0]
    df.drop(['label'],axis=1)

    data_mat = df.values
    train_data = data_mat[:,0:3]
    label = data_mat[:,-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    feature = min_max_scaler.fit_transform(train_data)
    return feature,label












       
     
        
        











