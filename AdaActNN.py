import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math




class AdaActANN:
    def __init__(self,a1=5,a2=2,n_features=1,n_classes=2,lr=0.0001):
        self.lr=lr=lr
        self.a1=a1
        self.a2=a2
        self.n_classes=n_classes
        self.n_features=n_features

        #Xavior Initialization
        self.W1=np.random.normal(0,math.sqrt(2/(self.n_features+self.a1)),(self.n_features,self.a1))
        self.W2=np.random.normal(0,math.sqrt(2/(self.a1+self.a2)),(self.a1,self.a2))
        self.W3=np.random.normal(0,math.sqrt(2/(self.a1+self.a2)),(self.a2,self.n_classes))
        self.b1=np.random.rand(self.a1)
        self.b2=np.random.rand(self.a2)
        self.b3=np.random.rand(self.n_classes)
        self.K=np.random.normal(0,1,(3,1))

    def g_function(self,x):
        return float(self.K[0])+float(self.K[1])*x

    def softmax(self,X):
        exps = np.exp(X - np.max(X,axis=1).reshape(-1,1))
        return exps / np.sum(exps,axis=1)[:,None]



    def forward(self,x):
        z1=np.dot(x,self.W1)+self.b1 #shape nxn_features x n_featuresxa1=nxa1
        a1=self.g_function(z1) #shape nxa1
        z2=np.dot(a1,self.W2)+self.b2 #shape nxa1 x a1xa2=nxa2
        a2=self.g_function(z2) #shape  nxa2
        z3=np.dot(a2,self.W3)+self.b3 #shape nxa2 x a2xn_classes=nxn_classes
        a3=self.softmax(np.array(z3)) #shape nxn_classes
        return z1,a1,z2,a2,z3,a3

    def backpropagate(self,x,labels):
        #Gradients
        t=x.shape[0]
        z1,a1,z2,a2,z3,a3=self.forward(x)
        dz3=a3-labels #shape nxn_classes
        dw3=(1/t)*np.dot(a2.T,dz3) #shape a2xn x nxn_classes=a2xn_classes
        db3=np.average(dw3,axis=0) #shape n_classes
        da2=np.dot(dz3,self.W3.T) #shape nxn_classes x n_classesxa2=nxa2
        k1=float(self.K[1])
        dz2=k1*da2 #shape nxa2
        dw2=(1/t)*np.dot(a1.T,dz2) #shape a2xn x nxa2=a2xa2
        db2=np.average(dw2,axis=0) #shape a2

        da2z2=np.multiply(np.squeeze(np.asarray(da2)),np.squeeze(np.asarray(z2)))
        da2z2z2=np.multiply(np.squeeze(np.asarray(da2)),np.multiply(np.squeeze(np.asarray(z2)),np.squeeze(np.asarray(z2))))
        dk2=np.array([np.mean(da2),np.mean(da2z2),np.mean(da2z2z2)]).reshape(3,1) #shape 3x1
        da1=np.dot(dz2,self.W2.T) #shape nxa2 x a2xa1=nxa1
        k2=float(self.K[1])
        dz1=k2*da1 #shape nxa1
        dw1=(1/t)*np.dot(x.T,dz1) #shape n_featuresxn x nxa1=n_featuesxa1
        db1=np.average(dz1,axis=0) #shape a1

        dk1=np.array([np.mean(np.squeeze(np.asarray(da1))),np.mean(np.squeeze(np.asarray(da1))*np.squeeze(np.asarray(z1))),
        np.mean(np.squeeze(np.asarray(da1))*np.squeeze(np.asarray(z1))*np.squeeze(np.asarray(z1)))]).reshape(3,1) #shape 3x1
        dk=dk2+dk1 #shape 3x1

        #Updation.
        self.W1=self.W1-self.lr*dw1
        self.b1=self.b1-self.lr*db1

        self.W2=self.W2-self.lr*dw2
        self.b2=self.b2-self.lr*db2

        self.W3=self.W3-self.lr*dw3
        self.b3=self.b3-self.lr*db3

        self.K=self.K-self.lr*dk

        return np.sum(dz3)










#Loading the data

data=pd.read_csv('iris.data')
X=data.iloc[:100,:-1].values
Y=data.iloc[:100,-1].values
Y=np.where(Y=='Iris-setosa',0,1)


#Splitting to 75% train and 25% test.
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
y_tc=y_test.copy()


#Performing normalization.
std_sc=StandardScaler()
std_sc.fit(X_train)

X_train=std_sc.transform(X_train)
X_test=std_sc.transform(X_test)



#One hot encoding
NN=AdaActANN(n_features=4)
oh_enc=OneHotEncoder(handle_unknown='ignore')
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
oh_enc.fit(y_train)
y_train=oh_enc.transform(y_train)
y_test=oh_enc.transform(y_test)





#Training for 1000 epochs.
loss=[]
epochs=[]
for ep in range(1000):

    lo=NN.backpropagate(X_train,y_train)
    loss.append(lo)
    epochs.append(ep)






print('Training Done')


#Prediction and calculating accuracy and precision.
_,_,_,_,_,y_pred=NN.forward(X_test)
y_pred=np.argmax(y_pred,axis=-1)
print(y_tc)
print(y_pred)
print(y_pred.shape)
print(f"Accuracy score={accuracy_score(y_tc,y_pred)}")
print(f"Precision={precision_score(y_tc,y_pred)}")

plt.plot(epochs,loss)
plt.show(block=False)
plt.savefig('Loss_epochs.jpg')
# plt.show()

