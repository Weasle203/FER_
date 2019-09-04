from PyQt5.QtWidgets import (QSplitter,QWidget,QLabel,QHBoxLayout,
                            QTextEdit,QFrame,QGridLayout,QApplication,
                            QStyleFactory,QVBoxLayout,QPushButton,QListView,QFileDialog)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage
import sys
import cv2
from modelLoader import findEmotion
class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setMinimumWidth(400)
        
class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setMinimumHeight(600)
class styledButton(QPushButton):
    def __init__(self,name):
        QPushButton.__init__(self,name)
        self.setMinimumWidth(180)
        self.setMaximumWidth(200)
        
class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.initializeVariables()
        self.UiSetup()
    def initializeVariables(self):
        self.camera = False
        self.faces = None
        self.videoMode = False
        self.processed = False
        self.captured = None
        self.emotion = ['Angry','Fear','Happy','Sad','Surprise','Neutral']#disgust has been removed
        #related to text of image
        self.font                   = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10,500)
        self.fontScale              = 1
        self.fontColor              = (255,255,255)
        self.lineType               = 2


    def UiSetup(self):
        self.original = QHBoxLayout()
        self.setLayout(self.original)
        #RIGHT SIDE KA CONSTRUCTION
        #RIGHT SIDE KA CONSTRUCTION
        self.rightLayout = QVBoxLayout()
        #constructing the label that keep the image or video
        self.imgLabel = QLabel()
        self.pixmap = QPixmap('Abstract.jpg')
        self.imgLabel.setMinimumWidth(400)
        self.imgLabel.setMinimumHeight(400)
        self.imgLabel.setMaximumWidth(800)
        self.imgLabel.setMaximumHeight(600)
        self.imgLabel.setScaledContents(True)
        self.imgLabel.setPixmap(self.pixmap)
        #setting the horizontal line separator
        self.hline = QHLine()
        
        #Bottom wala textArea
        self.textedit = QTextEdit()
        self.textedit.setMinimumWidth(400)
        self.textedit.setMinimumHeight(200)
        self.textedit.setMaximumWidth(800)
        self.textedit.setMaximumHeight(300)
        
        #right wala widget me Label(imgLabel) horizontal line and textedit dalna hai
        self.rightLayout.addWidget(self.imgLabel)
        self.rightLayout.addWidget(self.hline)
        self.rightLayout.addWidget(self.textedit)
        



        #LEFT SIDE KA CONSTRUCTION
        #LEFT SIDE KA CONSTRUCTION

        self.leftLayout = QVBoxLayout()
        self.button1 = styledButton("Open")
        self.button1.clicked.connect(self.button1_clicked)
        self.button2 = styledButton("Camera")
        self.button2.clicked.connect(self.button2_clicked)
        
        self.button3 = styledButton("VideoMode")
        self.button4 = styledButton("Process")
        self.button4.clicked.connect(self.button4_clicked)
        if self.faces != None:
            self.button4.setEnabled(True)
        else:
            self.button4.setEnabled(False)
        self.button5 = styledButton("Visulaize")
        
        if self.processed == True:
            self.button5.setEnabled(True)
        else:
            self.button5.setEnabled(False)
            
        

        #LEFT WALA ME WIDGET DALNA HAI    
        self.leftLayout.addWidget(self.button1)
        self.leftLayout.addWidget(self.button2)
        self.leftLayout.addWidget(self.button3)
        self.leftLayout.addWidget(self.button4)
        self.leftLayout.addWidget(self.button5)
        self.listView = QListView()
        self.listView.setMinimumHeight(400)
        self.listView.setMaximumWidth(200)
        self.leftLayout.addWidget(self.listView)
        
        
        # VERTICAL LINE SEPARATOR

        self.vline = QVLine()
        self.original.addLayout(self.leftLayout)
        self.original.addWidget(self.vline)
        self.original.addLayout(self.rightLayout)
        
        
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))
        self.setWindowTitle('FER')
    
    def recognise(self,img,face_cascade):
        self.faces = None
        self.captured =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.faces = face_cascade.detectMultiScale(self.captured)
        count = 0
        for x,y,l,b in self.faces:
            count += 1
            self.bottomLeftCornerOfText = (x,y)
            cv2.putText(img,'Face ' + str(count), 
                self.bottomLeftCornerOfText, 
                self.font, 
                self.fontScale,
                self.fontColor,
                self.lineType)
            cv2.rectangle(img,(x,y),(x+l,y+b),(0,225,255),1)
        if self.faces != []:
            return True
        else:
            return False
        
    
    def button1_clicked(self):
        options=QFileDialog.Options()
        filename,_ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()",
                                                 "","Images Files (*.png *.jpg)",
                                                 options=options)
        #file_dialog.selectNameFilter("Images (*.png *.jpg)")
        if filename:
            img = cv2.imread(filename)
            face_cascade= cv2.CascadeClassifier('cml/haarcascade_frontalface_default.xml')
            if self.recognise(img,face_cascade):
                self.button4.setEnabled(True)
                
            rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image,img.shape[1], img.shape[0],QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qimg)

            #myscaledpixmap = pixmap01.scaled(self.imgLabel.size(),Qt.KeepAspectRatio)
            self.imgLabel.setPixmap(pixmap01)        
            self.imgLabel.setScaledContents(True)
    def button2_clicked(self):
        self.cap = cv2.VideoCapture(0)
        self.textedit.setText('')
        if self.cap.isOpened():
            print("executed")
            self.camera = True
            self.button3.setEnabled(True)
            face_cascade= cv2.CascadeClassifier('cml/haarcascade_frontalface_default.xml')
        
        while self.camera:
            x = False
            ret,img = self.cap.read()
            img = cv2.flip(img,180)
            x = self.recognise(img,face_cascade)
            rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image,img.shape[1], img.shape[0],QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qimg)
            
            self.imgLabel.setPixmap(pixmap01)
            cv2.imshow('window',img)
            
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                self.image = img
                break
        if x:
            #if there is face in photo then it will process it otherwise it WON'T
            self.button4.setEnabled(True)

                
        if self.camera:
            cv2.destroyAllWindows()
            self.cap.release()
            self.button4.setEnabled(True)
        print("out of  loop")
        self.image = True
        
            
    def button3_clicked(self):
        pass
    
    def button4_clicked(self):
        if self.faces != [] :
            prediction = findEmotion(self.captured,self.faces)

            x_str = ""
            for c,i in enumerate(prediction):
                x_str += 'face ' + str(c) + '\n' 
                for j,k in enumerate(i[0]):
                    x_str += "{:<0} = {:^0.4f} \n".format(self.emotion[j],k)
                    print(k)
            self.textedit.setText(x_str)
            x_str = None
            prediction = None
            self.face = None
            self.captured = None
            self.button4.setEnabled(False)
            #here x will be list of observation corresponding to different faces in the

    def button5_clicked(self):
        pass
app = QApplication(sys.argv)
screen = Window()
screen.show()
app.exec_()