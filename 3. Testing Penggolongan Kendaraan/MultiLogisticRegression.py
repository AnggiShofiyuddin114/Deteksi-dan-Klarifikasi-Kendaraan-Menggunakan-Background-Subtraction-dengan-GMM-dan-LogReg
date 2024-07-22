import numpy as np
import cv2
class MultiLogisticRegression(object):
    def __init__(self,learning_rate = 0.3,max_iter = 100000,threshold = 0.001,progressBar = None):
        self.progressBar = progressBar
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.theta = None
    def fit(self,data,target):
        m,n=data.shape
        data1 = np.hstack((np.ones((m,1)),data))
        self.classes=np.unique(target)
        self.n_classes=len(self.classes)
        Y = np.zeros((m,self.n_classes))
        for cls in self.classes.astype(int):
            Y[np.where(target == cls),cls] = 1
        self.theta = np.ones((self.n_classes,n+1))
        for i in range(self.max_iter):
            if(self.progressBar!=None):
                self.progressBar.setValue((i+1)/self.max_iter*100)
            h=self.hyphothesis(data1)
            self.theta = self.gradient_descent(m,h,data1,Y)
            if(i!=0):
                CT2=self.CostTurunan(m,data1,Y)
                deltaCostTurunan=np.absolute(CT1-CT2)
                if(np.amax(deltaCostTurunan)<=self.threshold):
                    self.progressBar.setValue(100)
                    break
                CT1=CT2
            else:
                CT1=self.CostTurunan(m,data1,Y)
            key=cv2.waitKey(1)
            if key==27:
                break
        self.coef_=self.theta[:,1:]
        self.intercept_=self.theta[:,0]
    def predict(self,data):
        m,n=data.shape
        data1 = np.hstack((np.ones((m,1)),data))
        pred = np.zeros((m,self.n_classes))
        model_predict=self.hyphothesis(data1)
        return np.argmax(model_predict,axis=1)
    def CostTurunan(self,m,x,y):
        return np.dot(np.transpose([np.sum(x,axis=1)]),np.transpose([np.sum(y+m*(y-1)*np.exp(np.dot(x, self.theta.T)),axis=0)]).T)/(m*(np.exp(np.dot(x, self.theta.T))+1))
    def hyphothesis(self,x):
        z = np.dot(x, self.theta.T)
        return 1 / (1 + (np.exp(-z)))
    def gradient_descent(self,m,h,x,y):
        delta = (self.learning_rate/m) * np.dot((h-y).T, x)
        return self.theta - delta
