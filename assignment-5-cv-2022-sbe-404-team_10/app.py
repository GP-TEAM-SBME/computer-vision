import os
import cv2
import shutil
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui import Ui_MainWindow
from utils import *
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

clf_xml = 'haarcascade_frontalface_default.xml'
clf = cv2.CascadeClassifier(clf_xml)

def get_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img

base_pth = "./CroppedYale"
subjects_folders = os.listdir(base_pth)
print(len(subjects_folders))

# list of lists
subjects_imgs_names = []
# 38 subject 65 images each (some corrupted)
for subject in subjects_folders:
    allFiles = os.listdir(f"./{base_pth}/{subject}") 
    pgms = [ fname for fname in allFiles if fname.endswith('.pgm')]
    subjects_imgs_names.append(pgms)

unique_subject = []
for folder in subjects_imgs_names:
    unique_subject.append(folder[1])
    unique_subject.append(folder[2])
    unique_subject.append(folder[10])
    unique_subject.append(folder[15])
    unique_subject.append(folder[20])
    unique_subject.append(folder[25])
    unique_subject.append(folder[50])

unique_imgs_pth = []
for name in unique_subject:
    folder_name = name.split("_")[0]
    unique_imgs_pth.append(f"{folder_name}/{name}")
    
for i in range(len(unique_imgs_pth)):
    shutil.copy(f"{base_pth}/{unique_imgs_pth[i]}", f"unique_faces/")

subjects_imgs = os.listdir("./unique_faces")
len(subjects_imgs)

# 228 img 6 for each subject 38x6
recognition_pgms = []
for img in subjects_imgs:
        pgm = plt.imread(f"./unique_faces/{img}")
        recognition_pgms.append(pgm)

m = 192
n = 168 
# testing_pgms_arr
recognition_pgms_arr = np.array(recognition_pgms) #, dtype=object
# recognition_pgms_arr = np.array(testing_pgms)

recognition_pgms_vectors = []
for img in recognition_pgms_arr:
    img_flat = img.flatten()
    recognition_pgms_vectors.append(img_flat)

recognition_pgms_vectors = np.array(recognition_pgms_vectors) #, dtype=object

#  load avg_face
avg_face = np.loadtxt('avg_face.txt', dtype=int) #(32256,)

# Turn to numppy array of lists
vectors_list = []
for vector in recognition_pgms_vectors:
    vectors_list.append(vector.tolist())

# n_imgs x (nxm)
print("vector_list shape",np.array(vectors_list).shape)  

# (nxm) x k
vectors_list =  np.array(vectors_list).T 
print("vector_list.T shape",np.array(vectors_list).shape)   

# copy avg face n times
avg_face_list  = []
for i in range(len(vectors_list[1])):
    avg_face_list.append(avg_face.tolist())

avg_face_list =  np.array(avg_face_list).T

    
X_recognition = vectors_list - avg_face_list
print("X_recognition shape",X_recognition.shape)

U = np.loadtxt('U.txt')

def fit_svc_model(x_train, x_test, y_train, y_test):
    # svc = SVC(kernel='rbf', class_weight='balanced')
    print("TRAINING")
    svc = SVC(kernel='linear')
    model = make_pipeline(svc)
    
    param_grid = {'svc__C': [1],
                  'svc__gamma': [0.0001]}
    grid = GridSearchCV(model, param_grid)
    grid.fit(x_train, y_train)
    # print(grid.best_params_)
    model = grid.best_estimator_
    yfit = model.predict(x_test)
    return yfit, model

#  note:: from 15 to 39 labeled 14 to 38
def get_roc_data(start=5, end=7, data_shape=tuple):
    PCAmodes = [i for i in range(start,end)]

    counter = 0
    person_num = 1
    df_prev = pd.DataFrame()
    for i in range(data_shape[0]):
        PCA_P = U[ : , PCAmodes-np.ones_like(PCAmodes)].T @ X_recognition[ : ,    counter : counter+data_shape[1]]    
        counter += data_shape[1]
        df = pd.DataFrame(PCA_P.T)
        df['Person'] = person_num
        person_num +=1 
        df = pd.concat([df_prev,df], axis=0)
        df_prev = df 
        if person_num == data_shape[0]+1:
            break

    df_y = df.pop('Person')
    df_x = df    
    x_train,x_test,y_train,y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=40)
    yfit, model  = fit_svc_model(x_train,x_test,y_train,y_test)
    return y_test, yfit, model

def get_recognition_result(img_name, model, start=5, end=50):
    PCAmodes = [i for i in range(start,end)]
    img_idx = os.listdir('unique_faces').index(img_name)
    PCA_P = U[ : , PCAmodes-np.ones_like(PCAmodes)].T @ X_recognition[ : ,    img_idx]
    PCA_P = PCA_P.T
    return model.predict(PCA_P.reshape(1, -1))

def plot_roc_multiclass(y_test, y_pred, start,end):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    iter_list = [i for i in range(start,end+1)]
    for i in iter_list:
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in iter_list:
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()

    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


    return img
    


def get_df_data(start=5, end=7, data_shape=tuple):
    PCAmodes = [i for i in range(start,end)]

    counter = 0
    person_num = 1
    df_prev = pd.DataFrame()
    for i in range(data_shape[0]):
        PCA_P = U[ : , PCAmodes-np.ones_like(PCAmodes)].T @ X_recognition[ : ,    counter : counter+data_shape[1]]    
        counter += data_shape[1]
        df = pd.DataFrame(PCA_P.T)
        df['Person'] = person_num
        person_num +=1 
        df = pd.concat([df_prev,df], axis=0)
        df_prev = df 
        if person_num == data_shape[0]+1:
            break
        
    df_y = df.pop('Person')
    df_x = df
    return df_x, df_y

def get_roc_results(start, end):
    X, y = get_df_data(start,end, data_shape=(len(unique_imgs_pth),7))

    persons_classes = [i for i in range(1,39)]
    # Binarize the output
    y = label_binarize(y, classes=persons_classes)
    n_classes = y.shape[1]

    # # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                    random_state=40))
    print("############################")
    y_pred = classifier.fit(X_train, y_train).decision_function(X_test)
    print("############################")
    return plot_roc_multiclass(y_test, y_pred, start=start, end=end)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionOpen.triggered.connect(self.loadImg)
        self.ui.apply_detection.clicked.connect(self.apply_detection)
        self.ui.apply_recognition.clicked.connect(self.apply_recognition)
        self.img_path = None

    def loadImg(self):
        files_name = QtWidgets.QFileDialog.getOpenFileName( self, 'Open image', os.getenv('HOME'), "png(*)" )
        if len(files_name[0]) > 0:
            self.img_path = files_name[0]
            self.img_name = self.img_path.split('/')[-1]
            self.img = cv2.imread(self.img_path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img = cv2.resize(self.img, (300,300),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            showImage(self.ui.input_img, self.img)
        else:
            print("No file selected!")

    def apply_detection(self):
        img = get_detection(self.img)
        showImage(self.ui.output_image, img)

    def apply_recognition(self):
        start = int(self.ui.start_parameter.text())
        end = int(self.ui.end_parameter.text())
        y_test, yfit, model = get_roc_data(start, end, data_shape=(len(unique_imgs_pth),7))
        result = 'This person is yaleB' + str(get_recognition_result(self.img_name, model, start=start, end=end)[0])
        self.show_info_messagebox(result)
        # roc = plot_roc_multiclass(y_test, yfit, start=start, end=end)
        roc = get_roc_results(start=start, end=end)
        showImage(self.ui.output_image, roc)

    def show_info_messagebox(self, text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
    
        # setting message for Message Box
        msg.setText(text)
        
        # setting Message box window title
        msg.setWindowTitle("Recognition Result")
        
        # declaring buttons on Message Box
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        
        # start the app
        retval = msg.exec_()

    def closeEvent(self, event):
        choice = QtWidgets.QMessageBox.question(self, 'Message','Do you really want to exit?',QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
# function for launching a QApplication and running the ui and main window
def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    window()