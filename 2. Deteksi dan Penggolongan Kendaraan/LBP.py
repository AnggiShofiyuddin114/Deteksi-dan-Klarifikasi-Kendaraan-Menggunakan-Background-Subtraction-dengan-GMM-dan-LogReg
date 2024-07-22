import cv2
import numpy as np
class LBP(object):
    def __init__(self):
        koorTetanggaX=[0,1,1,1,0,-1,-1,-1]
        koorTetanggaY=[-1,-1,0,1,1,1,0,-1]
        a=np.ones((64,64))
        b=np.array([i for i in range(64)])
        koorKolom=np.array(a*b,dtype="int64")
        koorBaris=koorKolom.T
        self.koorX=np.transpose([koorKolom])+1+np.array(koorTetanggaX)
        self.koorY=np.transpose([koorBaris])+1+np.array(koorTetanggaY)
        self.pangkat=np.array([i for i in range(8)])
        self.points=[i for i in range(256)]
    def histogram(self,img):
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=self.img_resize(img,64,64)
        img1=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=[0,0,0])
        img_flat=img1.flatten()
        hasil=img_flat[self.koorY+self.koorX*(img.shape[0]+2)]-np.array(np.transpose([img.T]),dtype="int64")
        hasil_flat=np.where(hasil<0,0,1).reshape((1,img.shape[0]*img.shape[1],8))[0,:,:]
        hasil_akhir=np.sum(2**self.pangkat*hasil_flat,axis=1)
        lbp_flat=np.concatenate((hasil_akhir,self.points),axis=0)
        hist=np.unique(lbp_flat,return_counts=True)[1]-1
        return hist
    def img_resize(self,img,x,y):
        return cv2.resize(img,(y,x))
