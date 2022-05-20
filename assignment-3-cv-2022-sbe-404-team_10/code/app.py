# libraries needed for main python file
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, QObject
from utils import toQImage, mkdir_ifnotexists, showImage, createImageUiWidget
from gui import Ui_MainWindow
import sys
import cv2
from find_harris_corners import harris
import numpy as np

harrisSelect = False
siftSelect = False

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.images = []
        self.imageWidgets = []
        self.currImgIdx = -1

        self.imageWidgets.append(createImageUiWidget(self.ui))
        self.imageWidgets.append(createImageUiWidget(self.ui))

        # Link the UI components with the events
        self.ui.openAction.triggered.connect(self.loadImg)
        self.ui.newAction.triggered.connect(self.addWindow)
        self.ui.applyButton.clicked.connect(self.applyAlgorithm)
        self.ui.exitButton.clicked.connect(self.closeApplication)
        self.ui.harrisOption.toggled.connect(self.harrisSelected)
        self.ui.siftOption.toggled.connect(self.siftSelected)

    def loadImg(self):
        files_name = QtWidgets.QFileDialog.getOpenFileName( self, 'Open only jpeg, jpg, png', os.getenv('HOME'), "png(*.png)" )
        if len(files_name[0]) > 0:
            img_path = files_name[0]
            img_name = img_path.split('/')[-1].split('.')[0]
            self.currImgIdx = self.currImgIdx + 1
            img = cv2.imread(img_path)
            self.images.append(img)
            showImage(self.imageWidgets[self.currImgIdx], img)

        else:
            print("No file selected!")

    def harrisSelected(self, selected):
        if selected:
            print("harris selected")
            global harrisSelect
            global siftSelect
            harrisSelect = True
            siftSelect = False

    def siftSelected(self, selected):
        if selected:
            global harrisSelect
            global siftSelect
            harrisSelect = False
            siftSelect = True

    def applyAlgorithm(self):
        if harrisSelect == True:
            print("harris applied")
            img = harris(self.images[self.currImgIdx])
            showImage(self.ui.outputImage, img)
        if siftSelect == True:
            if self.currImgIdx:
                try:
                    from sift import sift
                    img = sift(self.images[self.currImgIdx], self.images[self.currImgIdx - 1])
                    showImage(self.ui.outputImage, img, size=(500, 250))
                except TypeError:
                    choice = QtWidgets.QMessageBox.warning(self, 'Message','This may take some minutes(unvectorized)...')
                    from template_matching_demo import sift
                    img = sift(self.images[self.currImgIdx], self.images[self.currImgIdx - 1])
                    showImage(self.ui.outputImage, img, size=(500, 250))

    def closeApplication(self):
        choice = QtWidgets.QMessageBox.question(self, 'Message','Do you really want to exit?',QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()
        else:
            pass
    
    def addWindow(self):
        newWindow()

def newWindow():
    global win
    win = MainWindow()
    win.show()

# function for launching a QApplication and running the ui and main window
def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    window()