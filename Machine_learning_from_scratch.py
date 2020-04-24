# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:46:26 2020

@author: User
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 

df =  pd.read_csv('SVM.csv')
"we dropped the unsufule values"
df.drop(['id','Unnamed: 32'],axis = 1 , inplace = True) 

" changing string value to int values"

df.diagnosis = [ 1 if each == 'M' else 0 for each in df.diagnosis ]

y = df.diagnosis

x = df.drop(['diagnosis'] , axis = 1)


train_x , test_x ,train_y,test_y = train_test_split(x,y,test_size = 0.2 , random_state = 40)

clf = SVC(kernel = 'rbf', gamma ='auto',verbose = True)

clf.fit(train_x , train_y)

bbb = clf.predict(test_x)

bab = clf.score(test_x,test_y)

#%%

print(__doc__)

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
# Loading some example data
from sklearn.ensemble import RandomForestClassifier

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])
clf5 = RandomForestClassifier()
clf1.fit(train_x, train_y)
clf2.fit(train_x,train_y )
clf3.fit(train_x,train_y )
eclf.fit(train_x,train_y )
clf5.fit(train_x,train_y)

print('Decision Tree example {}'.format(clf1.score(test_x,test_y)))
print('kneighborsClassifier {}'.format(clf2.score(test_x,test_y)))
print('SVC {}'.format(clf3.score(test_x,test_y)))
print('eclf {}'.format(eclf.score(test_x,test_y)))
print('clf5 {}'.format(clf5.score(test_x,test_y)))

#%%
"""
we will control the Hierarchy cluster
we create random number from normal value which has gaussian distrubition then we will plot it to see 
then we will concatenate it form numpy and at the end will make dictionary then form pd.Dataframe will make an pandas form then 
visualize it. 

"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)

x2 = np.random.normal(55,5,100)
y2 = np.random.normal(65,5,100)

x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {'x':x , 'y': y}

data = pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)

plt.show()
#%%
"""
from Scipy we will calculate the dendrogram of and according to the longest dedrogram will 

decide to have how many cluster  and the method was chosen is ward wich can be differ 


"""
from scipy.cluster.hierarchy import linkage ,dendrogram

merg = linkage(data,method = 'ward')

dendrogram(merg,leaf_rotation = 90)
plt.xlabel('data point')
plt.ylabel('euclidan distance')
plt.show()

#%%
"""
from agglomerative clustering we clarify the 3 cluseter the by ploting it we will 

see ther result.

"""
from sklearn.cluster import AgglomerativeClustering

hiyrachical_clustering = AgglomerativeClustering(n_clusters = 3 , affinity = 'euclidean',linkage = 'ward')

cluster = hiyrachical_clustering.fit_predict(data)
data['label'] = cluster

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color ='red')
plt.scatter(data.x[data.label == 1],data.y[data.label == 1], color = 'blue')
plt.scatter(data.x[data.label == 2],data.y[data.label == 2] , color = 'green')

plt.show()
