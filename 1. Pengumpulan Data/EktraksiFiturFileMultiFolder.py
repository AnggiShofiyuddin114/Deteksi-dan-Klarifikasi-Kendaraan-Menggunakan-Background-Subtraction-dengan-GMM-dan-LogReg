import cv2
import numpy as np
import sys
import os
import pandas as pd
from LBP import *
class EktraksiFiturFileMultiFolder(object):
    def __init__(self,ext=[]):
        self.ext = ext
        self.fileName=sys.argv[0].split("\\")[-1]
        self.lokasi=os.getcwd()
        self.run()
    def run(self):
        list_folder=os.listdir(self.lokasi)
        list_folder.remove("__pycache__")
        datasetComplete=[]
        targetComplete=[]
        kelas=0
        for folder in list_folder:
            if(len(folder.split("."))>1):
                continue
            filenameImage=[]
            selFolder=os.path.join(self.lokasi,folder)
            list_subFolder=os.listdir(selFolder)
            print("folder =",folder)
            dataset=[]
            for subFolder in list_subFolder:
                selFolder2=os.path.join(selFolder,subFolder)
                dirContent=os.listdir(selFolder2)
                print(subFolder)
                for each in dirContent:
                    selFile=os.path.join(selFolder2,each)
                    if(os.path.isfile(selFile) and self.cekFile(selFile)):
                        img=cv2.imread(selFile)
                        hasil=LBP().histogram(img)
                        dataset.append(hasil)
            if(len(datasetComplete)>0):
                datasetComplete=np.concatenate([datasetComplete,dataset],axis=0)
                targetComplete=np.concatenate([targetComplete,[kelas for i in range(len(dataset))]],axis=0)
            else:
                datasetComplete=dataset
                targetComplete=[kelas for i in range(len(dataset))]
            kelas+=1
        self.dataset=np.concatenate((datasetComplete,np.array([targetComplete]).T),axis=1)
        data_df=pd.DataFrame(self.dataset)
        data_df.to_csv(r'dataset.csv',index=False,header=False)
    def cekFile(self,file):
        cek=False
        for i in self.ext:
            if(file.endswith(i.upper()) or file.endswith(i.lower())):
                cek=True
                break
        return cek
app=EktraksiFiturFileMultiFolder(["jpg","png"])
