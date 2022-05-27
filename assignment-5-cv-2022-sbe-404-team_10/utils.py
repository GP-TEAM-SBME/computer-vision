from PyQt5.QtGui import QImage, qRgb, QPixmap
import numpy as np
import os
from pathlib import Path
import cv2
from PyQt5.QtCore import QThread, QObject
from PyQt5 import QtWidgets


gray_color_table = [qRgb(i, i, i) for i in range(256)]

def toQImage(im, copy=False):
    if im is None:
        return QImage()

    im = np.require(im, np.uint8, 'C')
    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim

def showImage(uiWidget, img, size=(300, 300)):
    img_resized = cv2.resize(img,size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    qimg = toQImage(img_resized, copy = True)
    pix = QPixmap(qimg)
    uiWidget.setPixmap(pix)