import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc




def prepare_images_pths(base_pth):
    ''' prepare imgs paths to be used in reading pgm imgs
    inputs
    ------
    base_pth: the parent directory containing subjects folders

    outputs
    -------
    subjects_imgs_pth: list of imgs_path
    '''

    subjects_folders = os.listdir(base_pth)
    subjects_imgs_names = []
    # 38 subject 65 images each (some corrupted)
    for subject in subjects_folders:
        allFiles = os.listdir(f"./{base_pth}/{subject}") 
        pgms = [ fname for fname in allFiles if fname.endswith('.pgm')]
        subjects_imgs_names.append(pgms)

    subjects_imgs_names_flatten = [item for sublist in subjects_imgs_names for item in sublist]



    subjects_imgs_pth = []
    for name in subjects_imgs_names_flatten:
        folder_name = name.split("_")[0]
        subjects_imgs_pth.append(f"{folder_name}/{name}")


    return subjects_imgs_pth, subjects_imgs_names



def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def plot_figure(image, label=str):
    plt.figure(figsize=(4., 4.))
    plt.title(f"{label}")
    plt.axis('off')   
    plt.imshow(image, plt.cm.gray)
    plt.show()




def get_unique_subjects(base_pth, subjects_imgs_names):
    ''' get unique subjects imgs to draw
    inputs
    ------
    subjects_imgs_names: list of images names from `prepare_images_pths`

    outputs
    -------
    unique_imgs_pgm
    '''
    unique_subject = []
    for folder in subjects_imgs_names[:36]:
        unique_subject.append(folder[1])

    unique_imgs_pth = []
    for name in unique_subject:
        folder_name = name.split("_")[0]
        unique_imgs_pth.append(f"{folder_name}/{name}")

    unique_imgs_pgm = []
    for i in range(len(unique_imgs_pth)):
        image = read_pgm(f"{base_pth}/{unique_imgs_pth[i]}", byteorder='<')
        unique_imgs_pgm.append(image)

    return unique_imgs_pgm        


def plot_imgs_mesh(imgs_mesh, n, m):
    fig = plt.figure(figsize=(12, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n, m),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, imgs_mesh):
        # Iterating over the grid returns the Axes.
        ax.axis('off')    
        ax.imshow(im, cmap="gray")

    plt.show()




def get_subject_pgms(base_pth, subjects_imgs_names, subject_num=0):
    # list of lists of yaleB01 imgs
    subject_imgs_pth = []
    for name in subjects_imgs_names[subject_num][:-1]: #except ambient img
        folder_name = name.split("_")[0]
        subject_imgs_pth.append(f"{folder_name}/{name}")

    subject_pgms = []
    for i in range(len(subject_imgs_pth)):
        image = read_pgm(f"{base_pth}/{subject_imgs_pth[i]}", byteorder='<')
        subject_pgms.append(image)

    return subject_pgms




def vectors_matrix(base_pth, start=0, end=36):

    subjects_folders = os.listdir(base_pth)
    len(subjects_folders)

    training_imgs_names = []
    for subject in subjects_folders[start:end]:
        allFiles = os.listdir(f"./{base_pth}/{subject}") 
        pgms = [ fname for fname in allFiles if fname.endswith('.pgm')]
        training_imgs_names.append(pgms)

    training_imgs_names_flatten = [item for sublist in training_imgs_names for item in sublist]

    training_imgs_pth = []
    for name in training_imgs_names_flatten:
        folder_name = name.split("_")[0]
        training_imgs_pth.append(f"{folder_name}/{name}")

    # more efficient pgm reading using matplot
    training_pgms = []
    for img in training_imgs_pth:
        pgm = plt.imread(f"./{base_pth}/{img}")
        training_pgms.append(pgm)

    training_pgms_arr = np.array(training_pgms) #, dtype=object

    training_pgms_vectors = []
    for img in training_pgms_arr:
        img_flat = img.flatten()
        training_pgms_vectors.append(img_flat)

    training_pgms_vectors = np.array(training_pgms_vectors) #, dtype=object     

    return training_pgms_vectors




def plot_reshape_vector(training_pgms_vectors, vector_num=5):
    # X = training_pgms_vectors >> reshape and vector to (192,168)
    m=192
    n=168
    img_reshaped = training_pgms_vectors[vector_num].reshape(m,n)
    plt.figure(figsize=(4., 4.))
    plt.axis('off')   
    plt.imshow(img_reshaped, plt.cm.gray)
    plt.show()    



def fix_corrupted_vectors(training_pgms_vectors):

    corrupted_imgs_idx = []
    for idx, vector in enumerate(training_pgms_vectors):
        if len(vector) == 307200:
            # print(f"{idx}")
            corrupted_imgs_idx.append(idx)


    fixed_training_pgms_vectors = np.delete(training_pgms_vectors, corrupted_imgs_idx)

    return fixed_training_pgms_vectors
        


def get_avg_face(fixed_training_pgms_vectors):

    vectors_list = []
    for vector in fixed_training_pgms_vectors:
        vectors_list.append(vector.tolist())
    
    vectors_list =  np.array(vectors_list).T # (nxm) x k
    avg_face = np.mean(vectors_list, axis=1)
    return avg_face



def X_matrix(base_pth, s_strat=0 , s_end=36  ):

    training_pgms_vectors = vectors_matrix(base_pth, start=s_strat, end=s_end)
    fixed_training_pgms_vectors = fix_corrupted_vectors(training_pgms_vectors)

    avg_face = get_avg_face(fixed_training_pgms_vectors)    

    avg_face_list  = []
    for i in range(fixed_training_pgms_vectors.shape[0]):
        avg_face_list.append(avg_face.tolist())

    avg_face_list =  np.array(avg_face_list).T
    
    vectors_list = []
    for vector in fixed_training_pgms_vectors:
            vectors_list.append(vector.tolist())

    vectors_list =  np.array(vectors_list).T        
    X = vectors_list - avg_face_list
    print("Matrix Shape: ", X.shape)

    return X
    

def svd_simultaneous_power_iteration(A, k, epsilon=0.00001):
    #source http://mlwiki.org/index.php/Power_Iteration
    n_orig, m_orig = A.shape
    if k is None:
        k=min(n_orig,m_orig)
        
    A_orig=A.copy()
    if n_orig > m_orig:
        A = A.T @ A
        n, m = A.shape
    elif n_orig < m_orig:
        A = A @ A.T
        n, m = A.shape
    else:
        n,m=n_orig, m_orig
        
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
 
    for i in range(1000):
        Z = A @ Q
        Q, R = np.linalg.qr(Z)
        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < epsilon:
            break
            
    singular_values=np.sqrt(np.diag(R))    
    if n_orig < m_orig: 
        left_vecs=Q.T
        #use property Values @ V = U.T@A => V=inv(Values)@U.T@A
        right_vecs=np.linalg.inv(np.diag(singular_values))@left_vecs.T@A_orig
    elif n_orig==m_orig:
        left_vecs=Q.T
        right_vecs=left_vecs
        singular_values=np.square(singular_values)
    else:
        right_vecs=Q.T
        #use property Values @ V = U.T@A => U=A@V@inv(Values)
        left_vecs=A_orig@ right_vecs.T @np.linalg.inv(np.diag(singular_values))

    return left_vecs, singular_values, right_vecs    


def PCA_projection(U_matrix, X_matrix, PCA_idx=[5,6,7]):
    PCA_P = U_matrix[ : , PCA_idx-np.ones_like(PCA_idx)].T @ X_matrix
    return PCA_P


def plot_cluster(PCA_P1, PCA_P2, pc1, pc2):
    '''
    pc1: idx of principal component from PCAmodes
    '''    
    plt.plot(PCA_P1[ pc1, : ], PCA_P1[ pc2, : ],'^',color='k',label='Person 38')
    plt.plot(PCA_P2[ pc1, : ], PCA_P2[ pc2, : ],'o',color='r',label='Person 39')
    plt.legend()
    plt.show()


def plot_confusion(y_test, yfit):
    mat = confusion_matrix(y_test, yfit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels= [0,1],
                yticklabels= [0,1])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
    

def fit_svc_model(x_train,x_test,y_train,y_test):
    
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(svc)
    
    param_grid = {'svc__C': [1, 5, 10, 50],
                  'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
    grid = GridSearchCV(model, param_grid)
    
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    
    model = grid.best_estimator_
    yfit = model.predict(x_test)
    
    return yfit


def get_roc_data(U_matrix, X_matrix, start=5, end=6, confusion=False): 
    PCAmodes = [i for i in range(start,end)]
    PCA_P1 = PCA_projection(U_matrix, X_matrix[ : ,    : 64], PCA_idx= PCAmodes)
    PCA_P2 = PCA_projection(U_matrix, X_matrix[ : , 64 :   ], PCA_idx= PCAmodes)

    df_38 = pd.DataFrame(PCA_P1.T)
    df_38['Person'] = 0
    df_39 = pd.DataFrame(PCA_P2.T)
    df_39['Person'] = 1
    df = pd.concat([df_38,df_39], axis=0)

    df_y = df.pop('Person')
    df_x = df
    
    x_train,x_test,y_train,y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=40)
    yfit = fit_svc_model(x_train,x_test,y_train,y_test)
    if confusion == True:
        plot_confusion(y_test, yfit)

    return y_test, yfit



    
def fit_svc_model_recognition(x_train, x_test, y_train, y_test):
    # svc = SVC(kernel='rbf', class_weight='balanced')
    svc = SVC(kernel='linear')
    model = make_pipeline(svc)
    
    param_grid = {'svc__C': [1, 5, 10, 50],
                  'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
    grid = GridSearchCV(model, param_grid)
    
    grid.fit(x_train, y_train)
    # print(grid.best_params_)
    
    model = grid.best_estimator_
    yfit = model.predict(x_test)
    return yfit

#  note:: from 15 to 39 labeled 14 to 38
def get_roc_data_recognition(U_matrix, X_matrix, start=5, end=7, data_shape=tuple):
    PCAmodes = [i for i in range(start,end)]
    counter = 0
    person_num = 1
    df_prev = pd.DataFrame()
    for i in range(data_shape[0]):
        PCA_P = U_matrix[ : , PCAmodes-np.ones_like(PCAmodes)].T @ X_matrix[ : ,    counter : counter+data_shape[1]]    
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
    yfit  = fit_svc_model_recognition(x_train,x_test,y_train,y_test)
    return y_test, yfit



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
    plt.figure()
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
    plt.show()
    


def get_df_dataframes(U_matrix, X_Matrix, start=5, end=7, data_shape=tuple):
    PCAmodes = [i for i in range(start,end)]

    counter = 0
    person_num = 1
    df_prev = pd.DataFrame()
    for i in range(data_shape[0]):
        PCA_P = U_matrix[ : , PCAmodes-np.ones_like(PCAmodes)].T @ X_Matrix[ : ,    counter : counter+data_shape[1]]    
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
