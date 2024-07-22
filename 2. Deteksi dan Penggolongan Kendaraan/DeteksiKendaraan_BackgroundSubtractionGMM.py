import cv2
import sys
import os
import shutil
import numpy as np
import time
from BackgroundSubtractionGMM import *
from MultiLogisticRegression import *
from LBP import *
import winsound
class DeteksiKendaraan_BackgroundSubtractionGMM(object):
    def __init__(self,Window):
        self.metode=MultiLogisticRegression()
        self.cekPenyimpanan()
        self.lokasi=os.getcwd()
        self.close=False
        self.view=True
        self.window=Window
        self.progressBar=Window.progressBar
        self.btn_mulai=Window.btn_mulai
        self.btn_batal=Window.btn_batal
        self.praStart=True
        self.pause=False
    def label(self,listObjekSF,listObjek,objek,no_id):
        bol_newId=True
        newObjek=True
        list_id=[]
        for i in listObjekSF:
            if(i[0][0]<=objek[1][0]<=i[0][1] and i[0][2]<=objek[1][1]<=i[0][3]):
                bol_newId= False
                no_id=i[-1]
                list_id.append([i[0],i[-1]])
        if(len(list_id)>1): 
            jarak_id=[]
            for i in list_id:
                jarak_id.append([self.jarak_ecludian(i[0],objek[0]),i[-1]])
            no_id=jarak_id[jarak_id.index(min(jarak_id))][-1]
        objek.append(no_id)
        listObjek.append(objek)
        return bol_newId,listObjek
    def run(self,video,background,backgroundBiner,ModelKlasifikasi,persen):
        self.nameFolder=video[video.rfind("/")+1:].replace(".avi","")
        self.createFolder(self.nameFolder)
        self.metode.theta=ModelKlasifikasi
        self.metode.n_classes=ModelKlasifikasi.shape[0]
        self.progressBar.setValue(0)
        self.persen=persen
        backgroundBiner = cv2.imread(backgroundBiner)
        self.backgroundBinerX=self.img_resize(backgroundBiner,persen)
        backgroundBiner=cv2.cvtColor(self.backgroundBinerX, cv2.COLOR_BGR2GRAY)
        backgroundBiner=np.where(backgroundBiner<128,0,backgroundBiner)
        self.backgroundBiner=np.where(backgroundBiner>=128,255,backgroundBiner)
        self.BackgroundSubtractionGMM=BackgroundSubtractionGMM(video,background,persen)
        no_id=1 
        listObjekSF=[]
        listObjekSS=[]
        id_kendaraan=[[1,4,0]]
##        Total=0
        ax,ay=self.backgroundBiner.shape
        tengahX=ax//2
        tengahY=ay//2
        waktu=time.ctime(time.time()).replace(" ","_").replace(":","-")
        self.nameOutput=video[video.rfind("/")+1:].replace(".","_"+waktu+".")
        lamaVideo=3*self.BackgroundSubtractionGMM.video.get(cv2.CAP_PROP_FRAME_COUNT)//self.BackgroundSubtractionGMM.video.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.nameOutput, cv2.VideoWriter_fourcc('M','J','P','G'),
                              (self.BackgroundSubtractionGMM.video.get(cv2.CAP_PROP_FRAME_COUNT)/self.BackgroundSubtractionGMM.skipframe-1)/lamaVideo,
                              (self.backgroundBiner.shape[1],self.backgroundBiner.shape[0]),1)
        metode=LBP()
        self.no_citra=1
        TotalGolKendaraan=[0,0,0,0]
        while True:
            key=cv2.waitKey(1)
            if key==27:
                break
            if(self.close or not self.window.bol_prosesDeteksi):
                out.release()
                break
            while not self.pause:
                try:
                    if(self.close):
                        self.pause=True
                        break
                    total=self.BackgroundSubtractionGMM.video.get(cv2.CAP_PROP_FRAME_COUNT)
                    pos=self.BackgroundSubtractionGMM.video.get(cv2.CAP_PROP_POS_FRAMES)
                    progres=((pos//4)/(total//4)*100//1)
                    self.progressBar.setValue(progres)
                    frameX,mask=self.BackgroundSubtractionGMM.gmm()
                    frameX2=frameX.copy()
                    cv2.rectangle(frameX,(0,tengahX*1),(ay-1,tengahX*2),(255,255,0),2)
                    mask=cv2.dilate(mask,None,iterations=4)
                    mask=cv2.erode(mask,None,iterations=4)
                    mask=cv2.dilate(mask,None,iterations=4)
                    if(self.praStart):
                        self.praStart=False
                        self.BackgroundSubtractionGMM.next()
                        continue
                    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    listObjek=[]
                    for contour in contours:
                        (x,y,z,h)=cv2.boundingRect(contour)
                        z-=1
                        h-=1
                        if(not (tengahX*1<(y+h//2)<tengahX*2)):
                            continue
                        if(cv2.contourArea(contour)<80*self.persen):
                            continue
                        if(self.backgroundBiner[y+h//2,x+z//2]==0):
                            continue
                        objek=[[y,y+h,x,x+z],np.array([y+h//2,x+z//2]),np.array([h,z]),h*z]
                        bol_newId,listObjek=self.label(listObjekSF,listObjek,objek,no_id)
                        if(bol_newId):
                            no_id+=1
                            id_kendaraan.append([no_id,4,0])
                    listObjekSF=listObjek
                    listObjekSS=self.seleksi(listObjekSF.copy(),ax,ay)
                    for i in listObjekSS:
                        citraTraining=True
                        margin=5
                        if(i[0][0]-margin>=0 and i[0][1]+margin<=270-1 and i[0][2]-margin>=0 and i[0][3]+margin<=480-1):
                            i[0][0]=i[0][0]-margin
                            i[0][1]=i[0][1]+margin
                            i[0][2]=i[0][2]-margin
                            i[0][3]=i[0][3]+margin
                        anggota=self.metode.predict(np.array([metode.histogram(frameX2[i[0][0]:i[0][1],i[0][2]:i[0][3]])]))
                        if(id_kendaraan[i[-1]-1][2] == 0):
##                            Total+=1
                            TotalGolKendaraan[anggota[0]]+=1
                            id_kendaraan[i[-1]-1][2]=1
                            citraTraining=False
                            self.saveImageObject(frameX2[i[0][0]:i[0][1],i[0][2]:i[0][3]], "Testing")
                        if(citraTraining): self.saveImageObject(frameX2[i[0][0]:i[0][1],i[0][2]:i[0][3]], "Training")
                        self.no_citra+=1
                        id_kendaraan[i[-1]-1][1]=anggota[0]
                        cv2.rectangle(frameX,(i[0][2],i[0][0]),(i[0][3],i[0][1]),(0,255,0),2)
                        cv2.putText(frameX,"ID    : "+str(i[-1]),(i[0][2]+6,i[0][0]+20),cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,0,255),2)
                        cv2.putText(frameX,"Class : "+str(id_kendaraan[i[-1]-1][1]),(i[0][2]+6,i[0][0]+40),cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,(0,0,255),2)
                    cv2.putText(frameX,"Golongan 0 : "+str(TotalGolKendaraan[0]),(10,20),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,(255,0,0),2)
                    cv2.putText(frameX,"Golongan 1 : "+str(TotalGolKendaraan[1]),(10,35),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,(255,0,0),2)
                    cv2.putText(frameX,"Golongan 2 : "+str(TotalGolKendaraan[2]),(10,50),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,(255,0,),2)
                    cv2.putText(frameX,"Golongan 3 : "+str(TotalGolKendaraan[3]),(10,65),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,(255,0,),2)
##                    cv2.putText(frameX,"Total Kendaraan : "+str(Total),(10,80),cv2.FONT_HERSHEY_SIMPLEX,
##                                    0.5,(255,0,0),2)
                    out.write(frameX)
                    if(self.view):
                        self.subplot4Image(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),frameX)
                    self.BackgroundSubtractionGMM.next()
                    key=cv2.waitKey(1)
                    if key==27:
                        self.pause=True
                        break
                except:
                    out.release()
                    self.pause=False
                    self.window.bol_prosesDeteksi=False
                    self.progressBar.setValue(100)
                    self.btn_batal.setEnabled(False)
                    self.btn_mulai.setText("Finish")
                    cv2.destroyAllWindows()
                    self.Beep()
                    break
    def img_resize(self,img,persen):
        x,y,z=img.shape
        return cv2.resize(img,(y*persen//100,x*persen//100))
    def jarak_ecludian(self,x,y):
        hasil=(np.array(x)-np.array(y))**2
        return np.sqrt(sum(hasil))
    def subplot4Image(self,img3,img4):
        color=[173,216,230]
        img1=cv2.copyMakeBorder(img3,35,10,10,10,cv2.BORDER_CONSTANT,value=color)
        cv2.putText(img1,"Video Mask",(int(21*8+10),25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        img2=cv2.copyMakeBorder(img4,35,10,10,10,cv2.BORDER_CONSTANT,value=color)
        cv2.putText(img2,"Hasil Deteksi",(int(21*8+1),25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        final_img=cv2.vconcat((img1,img2))
        cv2.imshow("Proses",final_img)
    def seleksi(self,listObjek,ax,ay):
        if(len(listObjek)>0):
            koorObjek=np.array(listObjek)[:,0]
            batas=np.array([0,ax-1,0,ay-1])
            uniqObjek=np.unique(np.where((np.array(list(koorObjek))==batas)==True)[0])
            for i in range(len(uniqObjek)-1,-1,-1):
                listObjek.pop(uniqObjek[i])
        return listObjek
    def createFolder(self,nameFolder):
        try:
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder)
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder+"\\Training")
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder+"\\Testing")
        except:
            shutil.rmtree(self.lokasi+"\\Save\\"+nameFolder)
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder)
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder+"\\Training")
            os.mkdir(self.lokasi+"\\Save\\"+nameFolder+"\\Testing")
    def saveImageObject(self,img,nameSubFolder):
        zero="0"*(6-len(str(self.no_citra)))
        cv2.imwrite(self.lokasi+"\\Save\\"+self.nameFolder+"\\"+nameSubFolder+"\\"+self.nameFolder+'_'+zero+str(self.no_citra)+".jpg",img)
##        self.no_citra+=1
    def cekPenyimpanan(self):
        list_folder=os.listdir(os.getcwd())
        hasil=len(np.where(np.array(list_folder)=='Save')[0])
        if(hasil==0):
            os.mkdir(os.getcwd()+"\\Save")
    def Beep(self):
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 2000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)
