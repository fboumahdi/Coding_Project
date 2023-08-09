#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b1fxb03

1) Unsupervised + supervised learning.
--------------------------------------

Attached is a data file dataClustering.csv which contains a data set of 2500 samples
with 8 features. 


i) Perform any clustering of your choice to determine the optimal # of clusters in the data


ii) Using the result of i) assign clusters labels to each sample, so each sample's label is the
cluster to which it belongs. 
Using these labels as the exact labels, you now have a labeled dataset.
	https://stats.stackexchange.com/questions/51418/assigning-class-labels-to-k-means-clusters

Build a classification model that classifies a sample with its corresponding label. 

Use multinomial regression as a benchmark model, and any ML model (trees, forests, SVM, NN etc.) as a comparison model.
Comment on which does better and why.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#%%

if sys.platform == "linux":
    directory = '/san/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding'
    
else:
    directory = "//rb/B1/NYRESAN/RDS/Work/nielsen/Projects/Fatima/Atlanta_Fed/quant_spec_coding"
    
chdir( directory )

#%%
#importing data
data_file = "data/dataClustering.csv"
data = pd.read_csv(data_file, header = None)

#determine the optimal number of clusters
inertia = []
for num_clusters in range(1,11):
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)
    
#plotting the elbow method graph 
plt.plot(range(1,11), inertia, marker = 'o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt

#%%
#so now we know the optimal number of clusters is 4 
num_clusters = 4

#applying KMeans clustering with the chosen number of clusters
kmeans = KMeans(n_clusters = num_clusters, random_state= 42)
kmeans.fit(data)

#assign cluster labels to each sample 
data['cluster_label'] = kmeans.labels_

#%%
#building the multinomial regression model 

#separate features and labels 
X = data.drop(['cluster_label'], axis = 1)
y = data['cluster_label']

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)  

#build multinomial logistic regression model
logreg = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', random_state = 42)
logreg.fit(X_train, y_train)

#make predictions
y_prediction_ml_reg = logreg.predict(X_test)

#calculate accuracy 
accuracy_ml_reg = accuracy_score(y_test, y_prediction_ml_reg)
print("Accuracy of Multinomial Logistic Regression: ", accuracy_ml_reg)

#%% building the comparison model (random forest)

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
random_forest.fit(X_train, y_train)

#make predictions
y_prediction_rforest = random_forest.predict(X_test)

#calculate accuracy
accuracy_rforest = accuracy_score(y_test, y_prediction_rforest)
print("Accuracy of Random Forest: ", accuracy_rforest)

#%% 
"""
Normally, a Random Forest Model would be the better model for the classification model
instead of the Multinomial Regression. Random Forest is better because it's less senstitive 
to outliers compared to Multinomial. Another reason why Random Forest is better is because it
has multiple trees where each tree is trained on a random subset which makes it better than using
a Trees model where overfitting is an issue. Since Multinomial Regression is also a single model, 
it's also prone to overfitting.
Since the cluster labels are kmeans, which should perfectly linearly searate the data
into a linear model, then we can achieve 100% accuracy.Despite changing the train/test
split to different combinations including 0.01/0.99, the accuracy is still 1.0. 
In this case, either model would work to predict 
the cluster labels accurately with this dataset.




























