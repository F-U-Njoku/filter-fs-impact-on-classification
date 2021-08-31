import warnings
import numpy as np
import pandas as pd
import openml as oml
from time import time
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, OneHotEncoder

from skfeature.function.information_theoretical_based import JMI

warnings.filterwarnings('ignore')

#Transformers and estimators
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
kfold = KFold(random_state=1, shuffle=True, n_splits=10)

nb_clf = MultinomialNB()
knn_clf = KNeighborsClassifier()
lda_clf = LinearDiscriminantAnalysis()
mlp_clf = MLPClassifier(random_state=1, alpha=0.0)

openml_datasets = [1015,1464,793,931,1021,40983,819,841,
1004,1061,41966,834,995,40666,41158,41145,
4340,1565,40496,1512,40498,1540,375,1526,
275,277,377,1518,41972,1560,1468,40499]

# TT --> Training time
print ("FS runtime, Base NB TT, 0.5 NB TT, Base NB accuracy, 0.5 NB accuracy,",
    "Base KNN TT, 0.5 KNN TT, Base KNN accuracy, 0.5 KNN accuracy,",
    "Base LDA TT, 0.5 LDA TT, Base LDA accuracy, 0.5 LDA accuracy,",
    "Base MLP TT, 0.5 MLP TT, Base MLP accuracy, 0.5 MLP accuracy")

for data in openml_datasets:
  dataset = oml.datasets.get_dataset(data)
  X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute)

  k= max(min((X.shape[0])/3,10),2) #Discretization
  num_fea5 = int(round((X.shape[1])**(0.5))) #Feature subset size

  disc = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
  X = disc.fit_transform(X.values.astype('float32'))
  y = le.fit_transform(y.values)

  fs_time = []  
  result = {'nb_clf':[[],[],[],[]],
            'knn_clf':[[],[],[],[]],
            'lda_clf':[[],[],[],[]],
            'mlp_clf':[[],[],[],[]]}

  for train_index, test_index in kfold.split(X):

    #Joint Mutual Information Feature selection
    start = time()
    idx,_,_ = JMI.jmi(X[train_index], y[train_index], n_selected_features=num_fea5)
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

      #with FS 0.5
      selected_features = X[:, idx[0:num_fea5]]
        
      start = time()
      clf[0].fit(selected_features[train_index], y[train_index])
      result[clf[1]][2].append(time()-start)
        
      y_pred = clf[0].predict(selected_features[test_index])  
      score = balanced_accuracy_score(y_pred, y[test_index])
      result[clf[1]][3].append(score)
    
  print (f"{round(np.median(fs_time),4)},",
  f"{round(np.median(result['nb_clf'][0]),4)},{round(np.median(result['nb_clf'][2]),4)},",
  f"{round(np.mean(result['nb_clf'][1]),4)},{round(np.mean(result['nb_clf'][3]),4)},",
  f"{round(np.median(result['knn_clf'][0]),4)},{round(np.median(result['knn_clf'][2]),4)},",
  f"{round(np.mean(result['knn_clf'][1]),4)},{round(np.mean(result['knn_clf'][3]),4)},",
  f"{round(np.median(result['lda_clf'][0]),4)},{round(np.median(result['lda_clf'][2]),4)},",
  f"{round(np.mean(result['lda_clf'][1]),4)},{round(np.mean(result['lda_clf'][3]),4)},",
  f"{round(np.median(result['mlp_clf'][0]),4)},{round(np.median(result['mlp_clf'][2]),4)},",
  f"{round(np.mean(result['mlp_clf'][1]),4)},{round(np.mean(result['mlp_clf'][3]),4)}")
