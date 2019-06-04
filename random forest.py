import pandas as pd
import numpy as np
import random
def raw_data_train_test(data,test_split):
    a=pd.read_csv(data,header=None)
#    a.iloc[:,1] = a.iloc[:,1].apply(lambda x: reutrn 0 if x=='M'  else return 1)
    for i in range(len(a)):
        if a.iloc[i,1]=="M":
            a.iloc[i,1]=0
        else:
            a.iloc[i,1]=1
    a=a.values
    np.random.shuffle(a)
    c=a[:,1]
    a=np.delete(a, np.s_[1], 1)
    a=np.column_stack([a,c])
    length=a.shape[0]
    d_train=a[0:(int(length*(1-0.2)))]
    d_test=a[length-(int(length*0.2)):length]
    return d_train, d_test
d_train, d_test =raw_data_train_test( r"http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",0.1)

def create_subsample(data,subsample_size_in_precent):
    subsample_data=([])
    list_indexes=list(range(0, len(data)))
    list_random_indexes=random.sample(list_indexes, int(len(data)*subsample_size_in_precent))
    for i in list_random_indexes:
        subsample_data=np.insert(subsample_data, 0, data[i], axis=0)
    subsample_data=subsample_data.reshape(int(len(subsample_data)/data.shape[1]),data.shape[1])
    return   subsample_data

def split_data(data,col_index,val):
    d_L=np.array([])
    d_R=np.array([])
    for i in range(len(data)):
        if val>=data[i,col_index]:
            d_L=np.insert(d_L, 0, data[i], axis=0)
        else:
            d_R=np.insert(d_R, 0, data[i], axis=0)
    d_L=d_L.reshape(int(len(d_L)/data.shape[1]),data.shape[1])
    d_R=d_R.reshape(int(len(d_R)/data.shape[1]),data.shape[1])
    return d_L, d_R

def impurity(data):
    temp1=0
    temp2=0
    if  len(data)==0 :
        impurity=1
    else:
        for i in range(len(data)):
             if data[i,data.shape[1]-1]==1:
                 temp1=temp1+1
             else:
                 temp2=temp2+1
             impurity  = 1-(((temp1/len(data))**2)+((temp2/len(data))**2))
    return impurity

def get_split(data):
    gini=100
    col_index=0
    val=0
    list_columns=list(range(0, data.shape[1]-2))
    list_random_columns=random.sample(list_columns, int(np.sqrt(data.shape[1]) ))
    for x in list_random_columns:
        for i in range(len(data)):
                d_L, d_R=split_data(data,x,data[i,x])
                d_L_impurity=impurity(d_L)
                d_R_impurity=impurity(d_R)
                avg=(d_L_impurity*(len(d_L)/len(data)))+(d_R_impurity*(len(d_R)/len(data)))
                if gini>avg:
                    gini=avg
                    col_index=x
                    val=data[i,x]
    return col_index, val 

def leaf(data):
    a=True
    for i in range(len(data)-1):
        if data[i,data.shape[1]-1]!=data[i+1,data.shape[1]-1]:
            a=False
            break
    return a

def lable_check(data):
    a=data[:,data.shape[1]-1].mean()
    if a>=0.5:
        return 1
    else:
        return 0
   
class Node:
    def __init__(self,data,max_depth):
        self.data=data
        self.max_depth=max_depth
        self.left_node=None
        self.right_node = None
        self.lable=None
        
    def build_tree(self):   
       if self.max_depth == 0 or leaf(self.data)==True:
           self.lable = lable_check(self.data)
           return  
                     
       self.col_index,self.val=get_split(self.data) 
       self.left_data,self.right_data=split_data(self.data,self.col_index,self.val)  
       
       self.left_node = Node(self.left_data,self.max_depth-1)
       self.left_node.build_tree()
       self.right_node = Node(self.right_data,self.max_depth-1)
       self.right_node.build_tree()
       
    def predict(self,test_row):
       if self.lable is not None:
           return self.lable 
       if test_row[:,self.col_index]<=self.val:
           return self.left_node.predict(test_row)        
       if test_row[:,self.col_index]>self.val:
           return self.right_node.predict(test_row)
        
def random_forest(train_data,subsample_size_in_precent,num_trees,tree_depth,test_data):
    trees = []
    prediction_matrix=np.zeros((num_trees,len(d_test)))
    for i in range(num_trees):
        subsample_data=create_subsample(train_data,subsample_size_in_precent)
        trees.append(Node(subsample_data,tree_depth))
        trees[i].build_tree()
    for x in range(len(test_data)):
        for i in range(num_trees) : 
            prediction_list=np.array([])
            prediction=trees[i].predict(test_data[x,:].reshape(1,32))
            prediction_list=np.append(prediction_list, prediction)
        prediction_matrix[:,x]=prediction_list            
    prediction_list_means=np.array([])
    for i in range(prediction_matrix.shape[1]):  
        if prediction_matrix[:,i].mean()>=0.5:
            prediction_list_means=np.append(prediction_list_means,1)
        else:
            prediction_list_means=np.append(prediction_list_means,0)        
    return prediction_list_means       

def Accuracy(predictions, test_data):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data[i,test_data.shape[1]-1]:
            correct += 1	
    Accuracy=((correct/(len(predictions))) * 100.0)
    return print(Accuracy)

prediction=random_forest(d_train,0.3,10,10,d_test)
Accuracy(prediction,d_test) 



