import cv2
import numpy as np
class BackgroundSubtractionGMM(object):
    def __init__(self,video,background,persen):
        self.skipframe=3
        self.video = cv2.VideoCapture(video)
        background = cv2.imread(background)
        self.backgroundX=self.img_resize(background,persen)
        background=self.normalisasi(self.backgroundX)
        self.background=self.gammaCorrection(background)
        _, frame=self.video.read()
        x,y,z=frame.shape
        frame=self.img_resize(frame,persen)
        self.frameX=frame.copy()
        frame=self.normalisasi(frame)
        self.frame=self.gammaCorrection(frame)
        self.data=abs(self.frame-self.background)
        self.persen = persen
        x,y,z=self.data.shape
        w=np.random.rand(x,y,3)
        m=np.mean(self.data,axis=2)
        s=np.std(self.data,axis=2)
        m=np.transpose([m.T])
        s=np.transpose([s.T])
        self.a = 0.1
        self.w,self.m,self.s=self.model_awal(w,m,s)
    def pencocokan_distribusi(self,data,m,s):
        batas_bawah = m-2.5*s
        batas_atas  = m+2.5*s
        hasilbb=np.array(batas_bawah<data,dtype="int32")
        hasilba=np.array(data<batas_atas,dtype="int32")
        hasil=np.where((hasilbb+hasilba)==2,True,False)
        hasil=np.unique(hasil,return_counts=True)
        hasilpd=False
        if ((np.array([hasil])==[1]).all(1).any()):
                hasilpd=True
        return hasilpd
    def model_awal(self,w,m,s):
        list_frame=[self.frame]
        for i in range(25):
            frame=np.sum(list_frame,axis=0)/len(list_frame)
            data=abs(frame-self.background)
            w,m,s=self.updateModel(self.a,m,s,data,w)
            hasil=self.pencocokan_distribusi(self.data,m,s)
            if(hasil):
                    break
            for i in range(self.skipframe):
                _,frame=self.video.read()
            frame=self.img_resize(frame,self.persen)
            self.frameX=frame.copy()
            frame=self.normalisasi(frame)
            self.frame=self.gammaCorrection(frame)
            list_frame.append(self.frame)
            key=cv2.waitKey(1)
            if key==27:
                break
        self.a=0.001
        return w,m,s
    def GaussianDistribusi(self,x,m,s):
        hasil=np.exp(-(np.transpose([np.transpose(np.sum((x-m)**2,axis=2))]))/(2*s))/(np.sqrt((44*s)/7))
        return hasil
    def M(self,x):
        m=np.mean(x,axis=2)
        s=np.std(x,axis=2)
        m=np.transpose([m.T])
        s=np.transpose([s.T])
        hasil=np.nan_to_num((x-m)/s)
        return np.where(hasil<=2.5,1,0)
    def updateModel(self,a,m,s,x,w):
        p=np.nan_to_num(a/w)
        #update weight
        hasil1=(1-a)*w+a*self.M(x)
        hasil1=self.normalisasi(hasil1)
        #update mean
        hasil2=(1-p)*m+p*x
        #update standar deviasi
        hasil3=np.sqrt((1-p)*s**2+p*(x-m)**2)
        return hasil1,hasil2,hasil3
    def B(self,GaussianD,T):
        GaussianD=np.where(GaussianD<T,0,GaussianD)
        hasil=np.where(GaussianD>=T,1,GaussianD)
        hasil=abs(1-hasil)
        return np.where(hasil==1,255,hasil)
    def gmm(self):
        frameX=self.frameX.copy()
        hasil=self.GaussianDistribusi(self.data,self.m,self.s)
        hasil=np.nan_to_num(hasil)
        hasil=np.array(self.B(hasil,1),dtype="uint8")
        x,y,z=hasil.shape
        if(z==3):
            hasil=cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)
            hasil=np.where(hasil<255,0,hasil)
            hasil=np.where(hasil>=255,255,hasil)
        return frameX,hasil
    def next(self):
        for i in range(self.skipframe):
            _,frame=self.video.read()
        frame=self.img_resize(frame,self.persen)
        self.frameX=frame.copy()
        frame=self.normalisasi(frame)
        self.frame=self.gammaCorrection(frame)
        self.data=abs(self.frame-self.background)
        self.w,self.m,self.s=self.updateModel(self.a,self.m,self.s,self.data,self.w)
    def img_resize(self,img,persen):
        x,y,z=img.shape
        return cv2.resize(img,(y*persen//100,x*persen//100))
    def normalisasi(self,x):
        min=np.amin(x)
        max=np.amax(x)
        return (x-min)/(max-min)*0.8+0.1
    def gammaCorrection(self,img):
        return 0.7*(img)**(1/2.2)
