import cv2
import numpy as np
import math
import time
import pandas as pd
import sys
import os
from PySide import QtGui,QtCore
from DeteksiKendaraan_BackgroundSubtractionGMM import *
from MultiLogisticRegression import *

class DeteksiKendaraan(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(DeteksiKendaraan, self).__init__(parent)
        self.no=1
        self.filenameDataset=None
        self.filenameBackground=None
        self.filenameVideo=None
        self.filenameBackgroundBiner=None
        self.bol_viewProses=True
        self.bol_train=False
        self.bol_prosesDeteksi=False
        self.bol_prosesKlasifikasi=False
        self.bol_dataset=False
        self.pos_page=0
        self.max_page=0
        self.split_page=100
        self.verticalHeaderLabel=np.array([i for i in range(1,self.split_page+1)])
        self.warna1=QtGui.QColor(0,255,0)
        self.warna2=QtGui.QColor(225,225,225)
        gambarKosong=np.zeros((108, 192,3),dtype="uint8")
        self.pixmap=self.convertToQtFormat(gambarKosong)
        self.retranslateUi()
    def retranslateUi(self):
        self.setWindowTitle('Deteksi Objek')
        self.resize(700, 520)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)

        self.btn_training = QtGui.QPushButton("Training Logistic Regression",self.gridLayoutWidget)
        self.btn_testing = QtGui.QPushButton("Deteksi Kendaraan",self.gridLayoutWidget)
        self.HLayout_0 = QtGui.QHBoxLayout()
        self.HLayout_0.addWidget(self.btn_training)
        self.HLayout_0.addWidget(self.btn_testing)
        self.gridLayout.addLayout(self.HLayout_0, 3, 0, 1, 3)
        #tampilan Training
        self.tampilanTraining()
        self.tampilanTesting()
        self.setCentralWidget(self.centralwidget)
        self.warnaButton()
        self.perintahButton()
        self.widgetTesting.setVisible(False)
        exit_action=QtGui.QAction("Exit",self)
        exit_action.triggered.connect(lambda:
                                      self.closeEvent(QtGui.QCloseEvent()))
    def tampilanTraining(self):
        self.widgetTraining = QtGui.QWidget()
        self.gridLayoutTraining = QtGui.QGridLayout(self.gridLayoutWidget)
        self.widgetTraining.setLayout(self.gridLayoutTraining)
        
        self.VLayout_1 = QtGui.QVBoxLayout()
        self.HLayout_1 = QtGui.QHBoxLayout()
        self.labelDataset = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Dataset Ektraksi Fitur Kendaraan</strong>",self)
        self.labelDataset.setAlignment(QtCore.Qt.AlignCenter)
        self.url_dataset = QtGui.QLineEdit(self.gridLayoutWidget)
        self.url_dataset.setMinimumWidth(560)
        self.btn_browseDataset = QtGui.QPushButton("Browse",self.gridLayoutWidget)
        self.VLayout_1.addWidget(self.labelDataset)
        self.HLayout_1.addWidget(self.url_dataset)
        self.HLayout_1.addWidget(self.btn_browseDataset)
        self.VLayout_1.addLayout(self.HLayout_1)
        self.gridLayoutTraining.addLayout(self.VLayout_1, 4, 0, 1, 3)
            
        self.HLayout_21 = QtGui.QHBoxLayout()
        self.HLayout_22 = QtGui.QHBoxLayout()
        self.HLayout_23 = QtGui.QHBoxLayout()
        self.labelLearningRate = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Learning Rate : </strong>",self)
        self.spinBox1 = QtGui.QDoubleSpinBox(self.gridLayoutWidget)
        self.spinBox1.setMaximumWidth(180)
        self.spinBox1.setRange(0.1,1)
        self.spinBox1.setSingleStep(0.1)
        self.spinBox1.setProperty("value", 0.3)
        self.HLayout_21.addWidget(self.labelLearningRate)
        self.HLayout_21.addWidget(self.spinBox1)
        self.labelIterasiMaksimal = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Iterasi Maksimal : </strong>",self)
        self.spinBox2 = QtGui.QSpinBox(self.gridLayoutWidget)
        self.spinBox2.setMaximumWidth(180)
        self.spinBox2.setRange(100000,500000)
        self.spinBox2.setSingleStep(10000)
        self.spinBox2.setProperty("value", 100000)
        self.HLayout_21.addWidget(self.labelIterasiMaksimal)
        self.HLayout_21.addWidget(self.spinBox2)
        self.labelThreshold = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Threshold : </strong>",self)
        self.spinBox3 = QtGui.QDoubleSpinBox(self.gridLayoutWidget)
        self.spinBox3.setMaximumWidth(180)
        self.spinBox3.setRange(0,1)
        self.spinBox3.setDecimals(3)
        self.spinBox3.setSingleStep(0.001)
        self.spinBox3.setProperty("value", 0.001)
        self.HLayout_21.addWidget(self.labelThreshold)
        self.HLayout_21.addWidget(self.spinBox3)
        self.checkBox = QtGui.QCheckBox(self.gridLayoutWidget)
        self.headerLabel = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Header Label</strong>",self)
        spacerItem = QtGui.QSpacerItem(0, 0, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.HLayout_22.addWidget(self.checkBox)
        self.HLayout_22.addWidget(self.headerLabel)
        self.HLayout_22.setContentsMargins(120,0,0,0)
        self.gridLayoutTraining.addLayout(self.HLayout_21, 5, 0, 1, 2)
        self.gridLayoutTraining.addLayout(self.HLayout_22, 5, 2, 1, 1)

        self.tableWidgetDataset = QtGui.QTableWidget(self.gridLayoutWidget)
        self.tableWidgetDataset.setFixedHeight(168)
        self.gridLayoutTraining.addWidget(self.tableWidgetDataset, 6, 0, 1, 3)
        
        self.page = QtGui.QComboBox(self)
        self.first = QtGui.QPushButton("First",self.gridLayoutWidget)
        self.last = QtGui.QPushButton("Last",self.gridLayoutWidget)
        self.previous = QtGui.QPushButton("Previous",self.gridLayoutWidget)
        self.next = QtGui.QPushButton("Next",self.gridLayoutWidget)
        self.HLayout_23.addWidget(self.first)
        self.HLayout_23.addWidget(self.previous)
        self.HLayout_23.addWidget(self.page)
        self.HLayout_23.addWidget(self.next)
        self.HLayout_23.addWidget(self.last)
        self.gridLayoutTraining.addLayout(self.HLayout_23, 7, 0, 1, 3)
        self.first.setEnabled(False)
        self.last.setEnabled(False)
        self.previous.setEnabled(False)
        self.next.setEnabled(False)
        
        self.progressBar2 = QtGui.QProgressBar(self)
        self.progressBar2.setValue(0)
        self.btn_mulai2 = QtGui.QPushButton("Start",self.gridLayoutWidget)
        self.gridLayoutTraining.addWidget(self.progressBar2, 8, 0, 1, 3)
        self.gridLayoutTraining.addWidget(self.btn_mulai2, 9, 0, 1, 3)
        
        self.labelHasil = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Hasil</strong>",self)
        self.labelHasil.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayoutTraining.addWidget(self.labelHasil, 10, 0, 1, 3)
        
        self.tableWidgetHasil = QtGui.QTableWidget(self.gridLayoutWidget)
        self.tableWidgetHasil.setFixedHeight(130)
        self.gridLayoutTraining.addWidget(self.tableWidgetHasil, 10, 0, 1, 3)
        self.gridLayout.addWidget(self.widgetTraining, 4, 0, 1, 3)
    def tampilanTesting(self):
        self.widgetTesting = QtGui.QWidget()
        self.gridLayoutTesting = QtGui.QGridLayout(self.gridLayoutWidget)
        self.widgetTesting.setLayout(self.gridLayoutTesting)
        spacerItem = QtGui.QSpacerItem(0, 65, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        
        self.VLayout_1 = QtGui.QVBoxLayout()
        self.HLayout_1 = QtGui.QHBoxLayout()
        self.gambarLabelBg = QtGui.QLabel(self)
        self.gambarLabelBg.setPixmap(self.pixmap)
        self.labelBg = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Background</strong>",self)
        self.labelBg.setAlignment(QtCore.Qt.AlignCenter)
        self.url_dataBg = QtGui.QLineEdit(self.gridLayoutWidget)
        self.url_dataBg.setMinimumWidth(80)
        self.btn_browseBg = QtGui.QPushButton("Browse",self.gridLayoutWidget)
        self.VLayout_1.addWidget(self.labelBg)
        self.HLayout_1.addWidget(self.url_dataBg)
        self.HLayout_1.addWidget(self.btn_browseBg)
        self.VLayout_1.addLayout(self.HLayout_1)
        self.VLayout_1.addItem(spacerItem)
        self.gridLayoutTesting.addWidget(self.gambarLabelBg, 4, 0, 1, 1)
        self.gridLayoutTesting.addLayout(self.VLayout_1, 4, 1, 1, 2)
        
        self.VLayout_2 = QtGui.QVBoxLayout()
        self.HLayout_2 = QtGui.QHBoxLayout()
        self.gambarLabelBb = QtGui.QLabel(self)
        self.gambarLabelBb.setPixmap(self.pixmap)
        self.labelBb = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Background Biner</strong>",self)
        self.labelBb.setAlignment(QtCore.Qt.AlignCenter)
        self.url_dataBb = QtGui.QLineEdit(self.gridLayoutWidget)
        self.url_dataBb.setMinimumWidth(80)
        self.btn_browseBb = QtGui.QPushButton("Browse",self.gridLayoutWidget)
        self.VLayout_2.addWidget(self.labelBb)
        self.HLayout_2.addWidget(self.url_dataBb)
        self.HLayout_2.addWidget(self.btn_browseBb)
        self.VLayout_2.addLayout(self.HLayout_2)
        self.VLayout_2.addItem(spacerItem)
        self.gridLayoutTesting.addWidget(self.gambarLabelBb, 5, 0, 1, 1)
        self.gridLayoutTesting.addLayout(self.VLayout_2, 5, 1, 1, 2)
        
        self.VLayout_3 = QtGui.QVBoxLayout()
        self.HLayout_3 = QtGui.QHBoxLayout()
        self.gambarLabelVideo = QtGui.QLabel(self)
        self.gambarLabelVideo.setPixmap(self.pixmap)
        self.labelVideo = QtGui.QLabel("<strong style='color: blue; font-family: Times-new-roman'>Video</strong>",self)
        self.labelVideo.setAlignment(QtCore.Qt.AlignCenter)
        self.url_dataVideo = QtGui.QLineEdit(self.gridLayoutWidget)
        self.url_dataVideo.setMinimumWidth(80)
        self.btn_browseVideo = QtGui.QPushButton("Browse",self.gridLayoutWidget)
        self.VLayout_3.addWidget(self.labelVideo)
        self.HLayout_3.addWidget(self.url_dataVideo)
        self.HLayout_3.addWidget(self.btn_browseVideo)
        self.VLayout_3.addLayout(self.HLayout_3)
        self.VLayout_3.addItem(spacerItem)
        self.gridLayoutTesting.addWidget(self.gambarLabelVideo, 6, 0, 1, 1)
        self.gridLayoutTesting.addLayout(self.VLayout_3, 6, 1, 1, 2)

        self.VLayout_4 = QtGui.QVBoxLayout()
        self.HLayout_4 = QtGui.QHBoxLayout()
        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setValue(0)
        self.btn_mulai = QtGui.QPushButton('Start', self)
        self.btn_batal = QtGui.QPushButton('Batal', self)
        self.btn_viewProses = QtGui.QPushButton('View Proses', self)
        self.btn_reset = QtGui.QPushButton('Reset', self)
        self.VLayout_4.addWidget(self.progressBar)
        self.HLayout_4.addWidget(self.btn_mulai)
        self.HLayout_4.addWidget(self.btn_batal)
        self.HLayout_4.addWidget(self.btn_viewProses)
        self.HLayout_4.addWidget(self.btn_reset)
        self.VLayout_4.addLayout(self.HLayout_4)
        self.gridLayoutTesting.addLayout(self.VLayout_4, 7, 0, 1, 3)
        self.gridLayout.addWidget(self.widgetTesting, 5, 0, 1, 3)
        self.btn_batal.setEnabled(False)
    def toggleBtn1(self):
        if(self.bol_train):
            self.btn_training.setStyleSheet("QWidget{background-color:%s}"%self.warna1.name())
            self.btn_testing.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
            self.bol_train = False
            self.widgetTesting.setVisible(False)
            self.widgetTraining.setVisible(True)
    def toggleBtn2(self):
        if(not self.bol_train):
            self.btn_training.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
            self.btn_testing.setStyleSheet("QWidget{background-color:%s}"%self.warna1.name())
            self.bol_train = True
            self.widgetTesting.setVisible(True)
            self.widgetTraining.setVisible(False)
    def warnaButton(self):
        self.btn_training.setStyleSheet("QWidget{background-color:%s}"%self.warna1.name())
        self.btn_testing.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
        self.btn_mulai.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
        self.btn_batal.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
        self.btn_viewProses.setStyleSheet("QWidget{background-color:%s}"%self.warna1.name())
        self.btn_reset.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
    def perintahButton(self):
        self.btn_training.clicked.connect(self.toggleBtn1)
        self.btn_testing.clicked.connect(self.toggleBtn2)
        self.btn_browseDataset.clicked.connect(self.openFileDataset)
        self.btn_browseBg.clicked.connect(self.openFileBg)
        self.btn_browseBb.clicked.connect(self.openFileBb)
        self.btn_browseVideo.clicked.connect(self.openFileVideo)
        self.btn_mulai.clicked.connect(self.mulaiDeteksi)
        self.btn_batal.clicked.connect(self.batal)
        self.btn_viewProses.clicked.connect(self.viewProses)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_mulai2.clicked.connect(self.mulaiKlasifikasi)
        self.first.clicked.connect(self.pageFirst)
        self.last.clicked.connect(self.pageLast)
        self.previous.clicked.connect(self.pagePrevious)
        self.next.clicked.connect(self.pageNext)
##        popupAboutToBeShown=QtCore.pyqtSignal()
        self.page.activated[str].connect(self.actionPage)
    
    def closeEvent(self,event):
        cv2.destroyAllWindows()
        event.accept()
    def openFileDataset(self):
        fileName, ok= QtGui.QFileDialog.getOpenFileName(self, "Open Text Files",
                                                 "c:/", "Open files(*.csv)")# ;; CSV(*.csv)")
        if(ok):
            self.url_dataset.setText(fileName)
            self.filenameDataset=fileName
            self.s_url_dataset=True
            self.getFileDataset(fileName)
    def getFileDataset(self,filename):
        try:
            # get data from csv
            self.tableWidgetDataset.clear()
            self.page.clear()
            self.pos_page=0
            self.tableWidgetHasil.clear()
            self.bol_dataset=True
            self.url_dataset.setText(filename)
            self.df = pd.read_csv(filename)
            self.dataset = np.array(self.df)
            self.n_data, self.m_data = self.df.shape
                
            # set data to tableWidget
            self.max_page=self.df.shape[0]/self.split_page
            if(self.max_page>int(self.max_page)):
                self.max_page=int(self.max_page)+1
            self.listPage=[str(i+1)+" / "+str(self.max_page) for i in range(self.max_page)]
            self.tableWidgetDataset.setColumnCount(self.df.shape[1])
            if(not self.checkBox.isChecked()):
                self.n_data+=1
                self.horizontalHeaderLabel=["X"+str(i) for i in range(1,self.m_data)]
                self.horizontalHeaderLabel.append("Y")
                header=[]
                for col in range(self.m_data):
                    item=self.df.columns[col]
                    if(item.count(".")==1):
                        item=item[:item.rfind(".")]
                        header.append(float(item))
                    else:
                        header.append(float(item))
                self.dataset=np.concatenate((np.array([header]),self.dataset),axis=0)
            self.getDataset()
            self.page.setEnabled(True)
            self.first.setEnabled(True)
            self.last.setEnabled(True)
            self.previous.setEnabled(True)
            self.next.setEnabled(True)
            self.page.addItems(self.listPage)
        except:
            self.tableWidgetDataset.clear()
            self.first.setEnabled(False)
            self.last.setEnabled(False)
            self.previous.setEnabled(False)
            self.next.setEnabled(False)
            self.warningDataset()
    def actionPage(self,text):
        index=text[:text.rfind(" / ")]
        self.pos_page=int(index)-1
        self.getDataset()
        print(index)
    def pageFirst(self):
        if(self.pos_page!=0):
            self.pos_page=0
            self.getDataset()
    def pageLast(self):
        if(self.pos_page!=self.max_page-1):
            self.pos_page=self.max_page-1
            self.getDataset()
    def pagePrevious(self):
        if(self.pos_page>0):
            self.pos_page-=1
            self.getDataset()
    def pageNext(self):
        if(self.pos_page<self.max_page-1):
            self.pos_page+=1
            self.getDataset()
    def getDataset(self):
        self.progressBar2.setValue(0)
        self.tableWidgetDataset.clear()
        self.tableWidgetDataset.setHorizontalHeaderLabels(np.array(self.horizontalHeaderLabel))
        if(self.pos_page<self.max_page-1 or self.n_data%self.split_page==0):
            batas_bawah=self.split_page*self.pos_page
            batas_atas=self.split_page*(self.pos_page+1)
            self.tableWidgetDataset.setRowCount(batas_atas-batas_bawah)
            verticalHeaderLabel=np.array(self.verticalHeaderLabel+batas_bawah,dtype="str")
            self.tableWidgetDataset.setVerticalHeaderLabels(verticalHeaderLabel)
            dataset=self.dataset[batas_bawah:batas_atas+1,:]
            for row in range(0,batas_atas-batas_bawah):
                for col in range(self.m_data):
                    self.tableWidgetDataset.setItem(row, col, QtGui.QTableWidgetItem(str(float(dataset[row, col]))))
                key=cv2.waitKey(1)
                if key==27:
                    break
        else:
            batas_bawah=self.split_page*(self.pos_page)
            batas_atas=self.n_data
            self.tableWidgetDataset.setRowCount(batas_atas-batas_bawah)
            verticalHeaderLabel=np.array(self.verticalHeaderLabel+batas_bawah,dtype="str")
            self.tableWidgetDataset.setVerticalHeaderLabels(verticalHeaderLabel)
            dataset=self.dataset[batas_bawah:batas_atas+1,:]
            for row in range(0,batas_atas-batas_bawah):
                for col in range(self.m_data):
                    self.tableWidgetDataset.setItem(row, col, QtGui.QTableWidgetItem(str(float(dataset[row, col]))))
                key=cv2.waitKey(1)
                if key==27:
                    break
        self.page.setCurrentIndex(self.pos_page)
    def openFileBg(self):
        fileName, ok= QtGui.QFileDialog.getOpenFileName(self, "Open Image Files",
                                                 "c:/", "*.jpg , *.jpeg , *.png")# ;; CSV(*.csv)")
        if(ok):
            unique_color=np.unique(cv2.imread(fileName))
            if(len(unique_color)>2):
                self.url_dataBg.setText(fileName)
                self.filenameBackground=fileName
                image=cv2.resize(cv2.imread(fileName),(192,108))
                self.showImageByPath(self.gambarLabelBg, image)
            elif(len(np.where((unique_color==[0,255])==False)[0])==0):
                self.warningBg()
            else:
                self.url_dataBg.setText(fileName)
                self.filenameBackground=fileName
                image=cv2.resize(cv2.imread(fileName),(192,108))
                self.showImageByPath(self.gambarLabelBg, image)
    def openFileBb(self):
        fileName, ok= QtGui.QFileDialog.getOpenFileName(self, "Open Image Files",
                                                 "c:/", "*.jpg , *.jpeg , *.png")# ;; CSV(*.csv)")
        if(ok):
            unique_color=np.unique(cv2.imread(fileName))
            if(len(unique_color)>2):
                self.warningBb()
            elif(len(np.where((unique_color==[0,255])==False)[0])==0):
                self.url_dataBb.setText(fileName)
                self.filenameBackgroundBiner=fileName
                image=cv2.resize(cv2.imread(fileName),(192,108))
                self.showImageByPath(self.gambarLabelBb, image)
            else:
                self.warningBb()
    def openFileVideo(self):
        fileName, ok= QtGui.QFileDialog.getOpenFileName(self, "Open Video Files",
                                                  "c:/", "*.mp4 , *.mkv , *.avi")
        if(ok):
            self.url_dataVideo.setText(fileName)
            self.filenameVideo=fileName
            video =cv2.VideoCapture(fileName)
            _, frame=video.read()
            image=cv2.resize(frame,(192,108))
            self.showImageByPath(self.gambarLabelVideo, image)
    def convertToQtFormat(self,image):
        convertToQtFormat=QtGui.QImage(image.data,image.shape[1],image.shape[0],QtGui.QImage.Format_RGB888)
        convertToQtFormat=QtGui.QPixmap.fromImage(convertToQtFormat)
        return QtGui.QPixmap(convertToQtFormat)
    def showImageByPath(self, label, image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pixmap=self.convertToQtFormat(rgbImage)
        label.setPixmap(pixmap)
    def mulaiKlasifikasi(self):
        if(self.bol_dataset):
            pilihan=QtGui.QMessageBox.question(self,self.tr("Konfirmasi"),self.tr("Apakah anda yakin ingin memulainya?"),
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.Yes)
            if(pilihan==QtGui.QMessageBox.Yes):   
                self.bol_prosesKlasifikasi=True
                dataset=self.normalisasi(self.dataset[:,:-1])
                self.metode=MultiLogisticRegression(learning_rate = self.spinBox1.value(),max_iter = self.spinBox2.value(),threshold = self.spinBox3.value(),progressBar = self.progressBar2)
                self.metode.fit(dataset,self.dataset[:,-1])
                np.save("ModelKlasifikasi",self.metode.theta)
                self.tableWidgetHasil.setColumnCount(self.metode.theta.shape[1])
                self.tableWidgetHasil.setRowCount(self.metode.theta.shape[0])
                for row in range(self.metode.theta.shape[0]):
                    for col in range(self.metode.theta.shape[1]):
                        self.tableWidgetHasil.setItem(row, col, QtGui.QTableWidgetItem(str(self.metode.theta[row, col])))
                self.n_classes=self.metode.n_classes 
                self.bol_prosesKlasifikasi=False
        else:
            self.warningDataset2()
    def normalisasi(self,data):
        min=np.transpose([np.amin(data,axis=0)]).T
        max=np.transpose([np.amax(data,axis=0)]).T
        hasil=(data-min)/(max-min)*0.8+0.1
        return hasil
    def mulaiDeteksi(self):
        if((self.filenameBackground!=None and self.filenameVideo!=None and self.filenameBackgroundBiner!=None) and self.btn_mulai.text()=="Start"
           and self.progressBar.value()==0):
            pilihan=QtGui.QMessageBox.question(self,self.tr("Konfirmasi"),self.tr("Apakah anda yakin ingin memulai deteksi kendaraan?"),
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.Yes)
            if(pilihan==QtGui.QMessageBox.Yes):
                try:
                    self.btn_mulai.setText("Jeda")
                    self.btn_batal.setEnabled(True)
                    self.bol_prosesDeteksi=True
                    self.deteksi=DeteksiKendaraan_BackgroundSubtractionGMM(self)
                    self.deteksi.run(self.filenameVideo,self.filenameBackground,self.filenameBackgroundBiner,np.load("ModelKlasifikasi.npy"),25)
                except:
                    self.btn_mulai.setText("Start")
                    self.btn_batal.setEnabled(False)
                    self.bol_prosesDeteksi=False
                    self.warningModelKlasifikasi()
        elif((self.btn_mulai.text()=="Start" or self.btn_mulai.text()=="Lanjutkan") and self.bol_prosesDeteksi):
            self.btn_mulai.setText("Jeda")
            self.deteksi.pause=False
        elif(self.btn_mulai.text()=="Jeda" and self.bol_prosesDeteksi):
            self.btn_mulai.setText("Lanjutkan")
            self.deteksi.pause=True
        elif(self.btn_mulai.text()=="Finish"):
            self.btn_mulai.setText("Start")
            self.progressBar.setValue(0)
            self.bol_prosesDeteksi=False
            self.actionReset()
        else:
            self.warning()
    def viewProses(self):
        if(self.bol_viewProses):
            self.bol_viewProses=False
            self.deteksi.view=False
            cv2.destroyAllWindows()
            self.btn_viewProses.setStyleSheet("QWidget{background-color:%s}"%self.warna2.name())
        else:
            self.bol_viewProses=True
            self.deteksi.view=True
            self.btn_viewProses.setStyleSheet("QWidget{background-color:%s}"%self.warna1.name())
    def reset(self):
        if((self.filenameBackground!=None or self.filenameVideo!=None or self.filenameBackgroundBiner!=None) and self.progressBar.value()==0 and not self.bol_prosesDeteksi):
            pilihan=QtGui.QMessageBox.question(self,self.tr("Konfirmasi"),self.tr("Apakah anda yakin ingin mereset?"),
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.Yes)
            if(pilihan==QtGui.QMessageBox.Yes):
                self.actionReset()
    def batal(self):
        pilihan=QtGui.QMessageBox.question(self,self.tr("Konfirmasi"),self.tr("Apakah anda yakin ingin membatalkan proses yang dilakukan?"),
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.Yes)
        if(pilihan==QtGui.QMessageBox.Yes):
            self.deteksi.close=True
            self.deteksi.pause=True
            cv2.destroyAllWindows()
            self.btn_mulai.setText("Start")
            self.btn_batal.setEnabled(False)
            self.progressBar.setValue(0)
            self.bol_prosesDeteksi=False
    def actionReset(self):
        self.url_dataBg.setText("")
        self.url_dataBb.setText("")
        self.url_dataVideo.setText("")
        self.filenameBackground=None
        self.filenameVideo=None
        self.filenameBackgroundBiner=None
        self.gambarLabelBg.setPixmap(self.pixmap)
        self.gambarLabelBb.setPixmap(self.pixmap)
        self.gambarLabelVideo.setPixmap(self.pixmap)
    def warning(self):
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Maaf, proses tidak dapat dimulai.\nMohon periksa kembali semua inputan."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def warningBb(self):
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Gambar Background Biner harus berwarna hitam putih."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def warningBg(self):
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Gambar background harus berwarna."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def warningDataset(self):
        self.tableWidgetDataset.clear()
        self.bol_dataset=False
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Dataset tidak valid."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def warningDataset2(self):
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Dataset belum dimasukan."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def warningModelKlasifikasi(self):
        QtGui.QMessageBox.warning(self,self.tr("Warning"),self.tr("Model klasifikasi belum dibuat."),
                                         QtGui.QMessageBox.Ok,QtGui.QMessageBox.Ok)
    def keyPressEvent(self, keyevent):
        event = keyevent.key()
        if event == QtCore.Qt.Key_Escape:
            self.close()
        if event == QtCore.Qt.Key_O:
            self.deteksi.pause=False
        if event == QtCore.Qt.Key_P:
            self.deteksi.pause=True
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = DeteksiKendaraan()
    myapp.show()
    sys.exit(app.exec_())
