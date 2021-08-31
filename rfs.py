import warnings
import numpy as np
import pandas as pd
import openml as oml
from time import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, OneHotEncoder

from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

warnings.filterwarnings('ignore')

#Transformers and estimators
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
kfold = KFold(random_state=1, shuffle=True, n_splits=10)

nb_clf = MultinomialNB()
knn_clf = KNeighborsClassifier()
lda_clf = LinearDiscriminantAnalysis()
mlp_clf = MLPClassifier(random_state=1, alpha=0.0)

openml_datasets = [841,1004,1061,41966,834,995,40666,41158,41145,
4340,1565,40496,1512,40498,1540,375,1526,
275,277,377,1518,41972,1560,1468,40499]
#1015,1464,793,931,1021,40983,819,

# TT --> Training time
print ("FS runtime, Base NB TT,0.4 NB TT, 0.5 NB TT, 0.6 NB TT, 0.7 NB TT, 0.8 NB TT, 0.9 NB TT,",
       "Base NB accuracy, 0.4 NB accuracy, 0.5 NB accuracy, 0.6 NB accuracy, 0.7 NB accuracy, 0.8 NB accuracy, 0.9 NB accuracy,",
       "Base KNN TT, 0.4 KNN TT, 0.5 KNN TT, 0.6 KNN TT, 0.7 KNN TT, 0.8 KNN TT, 0.9 KNN TT,",
       "Base KNN accuracy, 0.4 KNN accuracy, 0.5 KNN accuracy, 0.6 KNN accuracy, 0.7 KNN accuracy, 0.8 KNN accuracy, 0.9 KNN accuracy,",
       "Base LDA TT, 0.4 LDA TT, 0.5 LDA TT, 0.6 LDA TT, 0.7 LDA TT, 0.8 LDA TT, 0.9 LDA TT,",
       "Base LDA accuracy, 0.4 LDA accuracy, 0.5 LDA accuracy, 0.6 LDA accuracy, 0.7 LDA accuracy, 0.8 LDA accuracy, 0.9 LDA accuracy,",
       "Base MLP TT, 0.4 MLP TT, 0.5 MLP TT, 0.6 MLP TT, 0.7 MLP TT, 0.8 MLP TT, 0.9 MLP TT,",
       "Base MLP accuracy, 0.4 MLP accuracy, 0.5 MLP accuracy, 0.6 MLP accuracy, 0.7 MLP accuracy, 0.8 MLP accuracy, 0.9 MLP accuracy")

for data in openml_datasets:
  dataset = oml.datasets.get_dataset(data)
  X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute)

  k= max(min((X.shape[0])/3,10),2) #Discretization
  num_fea4 = int(round((X.shape[1])**(0.4))) #Feature subset size
  num_fea5 = int(round((X.shape[1])**(0.5))) #Feature subset size
  num_fea6 = int(round((X.shape[1])**(0.6))) #Feature subset size
  num_fea7 = int(round((X.shape[1])**(0.7))) #Feature subset size
  num_fea8 = int(round((X.shape[1])**(0.8))) #Feature subset size
  num_fea9 = int(round((X.shape[1])**(0.9))) #Feature subset size

  disc = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
  X = disc.fit_transform(X.values.astype('float32'))
  y = le.fit_transform(y.values)
  y_fs = construct_label_matrix(y)

  fs_time = []

  result = {'nb_clf':[[],[],[],[],[],[],
                      [],[],[],[],
                      [],[],[],[]],
            'knn_clf':[[],[],[],[],[],[],
                      [],[],[],[],
                      [],[],[],[]],
            'lda_clf':[[],[],[],[],[],[],
                      [],[],[],[],
                      [],[],[],[]],
            'mlp_clf':[[],[],[],[],[],[],
                      [],[],[],[],
                      [],[],[],[]]}

  for train_index, test_index in kfold.split(X):

    #Efficient and Robust Feature Selection Feature selection
    start = time()
    Weight = RFS.rfs(X[train_index], y_fs[train_index], gamma=0.1)
    idx = feature_ranking(Weight)
    fs_time.append(time()-start)

    for clf in [(nb_clf,'nb_clf'),(knn_clf,'knn_clf'), (lda_clf,'lda_clf'), (mlp_clf,'mlp_clf')]:
        #Classification
    
        #without FS
        start = time()
        clf[0].fit(X[train_index], y[train_index])
        result[clf[1]][0].append(time()-start)
        
        y_pred = clf[0].predict(X[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][1].append(score)

        
        #with FS 0.4
        selected_features = X[:, idx[0:num_fea4]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][2].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][3].append(score)
        #with FS 0.5
        selected_features = X[:, idx[0:num_fea5]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][4].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][5].append(score)
        
        #with FS 0.6
        selected_features = X[:, idx[0:num_fea6]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][6].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][7].append(score)
        
        #with FS 0.7
        selected_features = X[:, idx[0:num_fea7]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][8].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][9].append(score)
        
        #with FS 0.8
        selected_features = X[:, idx[0:num_fea8]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][10].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][11].append(score)
        
        #with FS 0.9
        selected_features = X[:, idx[0:num_fea9]]
        
        start = time()
        clf[0].fit(selected_features[train_index], y[train_index])
        result[clf[1]][12].append(time()-start)
        
        y_pred = clf[0].predict(selected_features[test_index])
        score = balanced_accuracy_score(y_pred, y[test_index])
        result[clf[1]][13].append(score)

  #Results
  print (f"{round(np.median(fs_time),4)},",
  f"{round(np.median(result['nb_clf'][0]),4)},{round(np.median(result['nb_clf'][2]),4)},{round(np.median(result['nb_clf'][4]),4)},{round(np.median(result['nb_clf'][6]),4)},{round(np.median(result['nb_clf'][8]),4)},{round(np.median(result['nb_clf'][10]),4)},{round(np.median(result['nb_clf'][12]),4)},",
  f"{round(np.mean(result['nb_clf'][1]),4)},{round(np.mean(result['nb_clf'][3]),4)},{round(np.mean(result['nb_clf'][5]),4)},{round(np.mean(result['nb_clf'][7]),4)},{round(np.mean(result['nb_clf'][9]),4)},{round(np.mean(result['nb_clf'][11]),4)},{round(np.mean(result['nb_clf'][13]),4)},",
  f"{round(np.median(result['knn_clf'][0]),4)},{round(np.median(result['knn_clf'][2]),4)},{round(np.median(result['knn_clf'][4]),4)},{round(np.median(result['knn_clf'][6]),4)},{round(np.median(result['knn_clf'][8]),4)},{round(np.median(result['knn_clf'][10]),4)},{round(np.median(result['knn_clf'][12]),4)},",
  f"{round(np.mean(result['knn_clf'][1]),4)},{round(np.mean(result['knn_clf'][3]),4)},{round(np.mean(result['knn_clf'][5]),4)},{round(np.mean(result['knn_clf'][7]),4)},{round(np.mean(result['knn_clf'][9]),4)},{round(np.mean(result['knn_clf'][11]),4)},{round(np.mean(result['knn_clf'][13]),4)},",
  f"{round(np.median(result['lda_clf'][0]),4)},{round(np.median(result['lda_clf'][2]),4)},{round(np.median(result['lda_clf'][4]),4)},{round(np.median(result['lda_clf'][6]),4)},{round(np.median(result['lda_clf'][8]),4)},{round(np.median(result['lda_clf'][10]),4)},{round(np.median(result['lda_clf'][12]),4)},",
  f"{round(np.mean(result['lda_clf'][1]),4)},{round(np.mean(result['lda_clf'][3]),4)},{round(np.mean(result['lda_clf'][5]),4)},{round(np.mean(result['lda_clf'][7]),4)},{round(np.mean(result['lda_clf'][9]),4)},{round(np.mean(result['lda_clf'][11]),4)},{round(np.mean(result['lda_clf'][13]),4)},",
  f"{round(np.median(result['mlp_clf'][0]),4)},{round(np.median(result['mlp_clf'][2]),4)},{round(np.median(result['mlp_clf'][4]),4)},{round(np.median(result['mlp_clf'][6]),4)},{round(np.median(result['mlp_clf'][8]),4)},{round(np.median(result['mlp_clf'][10]),4)},{round(np.median(result['mlp_clf'][12]),4)},",
  f"{round(np.mean(result['mlp_clf'][1]),4)},{round(np.mean(result['mlp_clf'][3]),4)},{round(np.mean(result['mlp_clf'][5]),4)},{round(np.mean(result['mlp_clf'][7]),4)},{round(np.mean(result['mlp_clf'][9]),4)},{round(np.mean(result['mlp_clf'][11]),4)},{round(np.mean(result['mlp_clf'][13]),4)}")
