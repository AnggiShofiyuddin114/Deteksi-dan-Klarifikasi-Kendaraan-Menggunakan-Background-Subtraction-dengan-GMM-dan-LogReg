import cv2
import numpy as np
import sys
import os
import pandas as pd
from csv import reader
from LBP import *
from MultiLogisticRegression import *
from sklearn.metrics import confusion_matrix
import winsound
class akurasiGolKendaraan(object):
    def __init__(self,ModelKlasifikasi):
        self.fileName=sys.argv[0].split("\\")[-1]
        self.lokasi=os.getcwd()
        self.metode=MultiLogisticRegression()
        self.metode.theta=ModelKlasifikasi
        self.metode.n_classes=self.metode.theta.shape[0]
        self.precRec=[]
        self.NamaFolder=[]
        self.run()
    def run(self):
        list_folder=os.listdir(self.lokasi)
        list_folder.remove("__pycache__")
        ListPrecRec=[]
        ListAkurasiRata2=[]
        ListAkurasi=[]
        KAkurasi=[]
        for folder in list_folder:
            if(len(folder.split("."))>1): continue
            self.NamaFolder.append(folder)
            selFolder=os.path.join(self.lokasi,folder)
            list_subFolder=os.listdir(selFolder)
            print("folder =",folder)
            targetTesting=[]
            targetPred=[]
            kelas=0
            for subFolder in list_subFolder:
                selFolder2=os.path.join(selFolder,subFolder)
                dirContent=os.listdir(selFolder2)
                tarPredClass=[]
                for each in dirContent:
                    selFile=os.path.join(selFolder2,each)
                    hasil=LBP().histogram(cv2.imread(selFile))
                    tarPredClass.append(self.metode.predict(np.array([hasil]))[0])
                if(len(targetTesting)>0):
                    targetTesting=np.concatenate([targetTesting, [kelas for i in range(len(dirContent))]], axis=0)
                    targetPred=np.concatenate([targetPred, tarPredClass], axis=0)
                else:
                    targetTesting=[kelas for i in range(len(dirContent))]
                    targetPred=tarPredClass
                kelas+=1
            self.conf=confusion_matrix(targetTesting, targetPred)
            print(self.conf)
            KAkurasi.append(self.KlasifikasiAkurasi(self.conf))
            self.list_pr=[]
            for i in range(len(np.unique(targetTesting))):
                pr=self.precision_recall(confusion_matrix(np.where(targetTesting!=i,1,0), np.where(targetPred!=i,1,0)))
                self.list_pr.append(pr)
            self.list_pr=np.array(self.list_pr)
            p=self.list_pr[:,0]
            r=self.list_pr[:,1]
            akurasi=np.nan_to_num(2*(r*p)/(r+p))
            ListPrecRec.append(self.list_pr.mean(axis=0)*100)
            ListAkurasiRata2.append(np.around(akurasi*100,2))
            ListAkurasi.append(np.mean(akurasi)*100)
        ListAkurasiRata2V2=[]
        for akurasi in ListAkurasiRata2:
            for i in range(4-len(akurasi)):
                akurasi=np.concatenate((akurasi,[-1]))
            ListAkurasiRata2V2.append(akurasi)
        ListAkurasiRata2V2=(np.array(ListAkurasiRata2V2,dtype="str"))
        posStrip=np.where(ListAkurasiRata2V2=="-1.0")
        ListAkurasiRata2V2[posStrip]=np.array(["-" for i in range(len(posStrip[0]))])
        hasil1=np.concatenate((ListAkurasiRata2V2,np.around(ListPrecRec,2),np.transpose([np.around(ListAkurasi,2)]),np.transpose([np.around(KAkurasi,2)])),axis=1)
        jumNilKes=np.array([0 for i in range(len(hasil1[0]))],dtype="float64")
        NNilKes=np.array([0 for i in range(len(hasil1[0]))])
        jumNilPos=np.array([[0 for i in range(len(hasil1[0]))] for i in range(3)],dtype="float64")
        NNilPerPos=np.array([[0 for i in range(len(hasil1[0]))] for i in range(3)])
        no=0
        for baris in np.array(hasil1):
            index=np.where(baris!='-')
            jumNilKes[index]+=np.array(baris[index],dtype="float64")
            NNilKes[index]+=np.array([1 for i in range(len(index[0]))])
            jumNilPos[no%3][index]+=np.array(baris[index],dtype="float64")
            NNilPerPos[no%3][index]+=np.array([1 for i in range(len(index[0]))])
            no+=1
        AkurasiPos=jumNilPos/NNilPerPos
        AkurasiKes=jumNilKes/NNilKes
        hasil=np.concatenate((hasil1,np.around(AkurasiPos,2),np.around([AkurasiKes],2)),axis=0)
        session=["session_center","session_left","session_right"]
        print(hasil)
        self.convCSV("Akurasi",hasil,_index=np.concatenate((self.NamaFolder,["Akurasi rata "+session[i] for i in range(3)],["Akurasi rata"])),
                     _header=np.concatenate((["Nama Video"],["F1-Score Class "+str(i) for i in range(4)],["Precision","Recall","F1 Score","Classification Accuracy"])))
    def KlasifikasiAkurasi(self,conf):
        diagonal=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        n_class=conf.shape[0]
        diagonal=diagonal[0:n_class]
        akurasi=np.sum(conf[np.where(diagonal==1)])/np.sum(conf)
        return akurasi*100
    def precision_recall(self,mat):
        precision=mat[0,0]/(mat[0,0]+mat[1,0])
        recall=mat[0,0]/(mat[0,0]+mat[0,1])
        return np.nan_to_num([precision,recall])
    def convCSV(self, name, dataset, _index=[], _header=[]):
        if(len(_index)==0 and len(_header)!=0):
            dataset=np.concatenate(([_header], dataset), axis=0)
        elif(len(_index)!=0 and len(_header)==0):
            dataset=np.concatenate((np.transpose([_index]), dataset), axis=1)
        elif(len(_index)!=0 and len(_header)!=0):
            dataset=np.concatenate((np.transpose([_index]), dataset), axis=1)
            dataset=np.concatenate(([_header], dataset), axis=0)
        data_df=pd.DataFrame(dataset)
        data_df.to_csv(r''+name+'.csv', index=False, header=False)
metode=MultiLogisticRegression()
metode.theta=np.load("ModelKlasifikasi.npy")
app=akurasiGolKendaraan(metode.theta)
