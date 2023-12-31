# -*- coding: utf-8 -*-
"""Final_ML_2.0_30FTs_94%

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/112azt06aCJihMZJLHkR8uVfPdUxSR0nT

Autonomous University of Chihuahua

Engineeing Faculty

Machine Learning

Assignment : Project Advances 2.2 - Explore and preparation of Database; 2.3 Logistic Regression Implementation

Professor: M.A. Olanda Prieto

Eng. Daniela Alejandra Rubio Vega

MIC P372953

For this project, it is used the Wisconcin's Breast Cancer dataset. It has over 569 instances, with 33 features each. The goal is to classify with the features if the diagnosis of the tumor is Benign (B) or Malignant (M).

From the 569 cases, there are 357 B and 212 M
"""

#Libraries needed are imported

import pandas as pd #data manipulation and analysis
import numpy as np #numerical operations with python
import matplotlib.pyplot as plt #graphs
import seaborn as sns #statistical graphs
import os #for interaction with operating system

#This block is used to ignore all the warning messages that may appear
import warnings
warnings.filterwarnings('ignore')

"""1. Prepare data set"""

#The variable for the dataset path is created and then printed
dirname = '/content/drive/MyDrive/breast_cancer.csv'

print(os.path.join(dirname)) #and then printed

#Read the csv file and making it into a pandas DataFrame to be manipulated
df = pd.read_csv('/content/drive/MyDrive/breast_cancer.csv')

#Show the first rows of the DF and their columns
df.head()

#It shows the dimensions of the DF created
df.shape #The tuple is the number of rows, and numver of columns

#It displays the statitics summary for each column of the DF
df.describe().T

#It is used to retrieve and print the unique values existing in the column of diagnosis
df.diagnosis.unique()

#It counts how many times each value of diagnosis is present
df['diagnosis'].value_counts()

##1.1-Clean & Prepare data

#Code used to remove the columns id & Unnamed from the DF, due that are not useful in the data analysis
df.drop('id', axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)

#Show the first rows of the DF and their columns
df.head()

#This code is used to select the diagnosis column, and in there, replace the values 'M' and 'B' with '1' and '0' respectively
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head() ##Show the first rows of the DF and their columns

#This is used to count in all the columns the missing values of each
df.isnull().sum()

"""As it is shown above, there are no missing values in the dataset"""

#This line computes the correlation matrix for the DF
df.corr()

"""The only correlations that will be taken in consideration are those that have a medium to strong (from 0.5 to 1.0) positive or negative relation. So the variables texture_mean, texture_se, smoothness_mean, smoothness_se, compactness_se, concavity_se, symmetry_mean, fractal_dimension_mean, texture_se, texture_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, texture_worst, smoothness_worst, symmetry_worst, and fractal_dimension_worst will be eliminated later."""

plt.hist(df['diagnosis'], color='g') #Make a histogram of the diagnosis column in color green
plt.title('Plot_Diagnosis (M=1 , B=0)') #This is to make the grpah's title
plt.show() #Display the graph

"""2. Data augmentation and transformation for databases with images. If you do not have images, analyze relevant characteristics that allow you to identify a better evaluation in the data."""

#The next list is created to store the names of the columns that will be elimnated (as mentioned before)
cols_to_drop = ['texture_mean','smoothness_se','smoothness_mean', 'compactness_se', 'concavity_se', 'symmetry_mean', 'fractal_dimension_mean',
                'texture_se', 'texture_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                'texture_worst', 'smoothness_worst', 'symmetry_worst', 'fractal_dimension_worst']
df.drop(cols_to_drop, axis=1, inplace=True) #Here the features nonuseful are dropped from the DF

#See the correlation matrix for the DF
df.corr()

"""As shown in the correlation matrix, there are only medium to strong relations now"""

#This block creates a 20 x 20 heatmap of the correlation matrix, with its values inside of the cells
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)

"""With the heatmap, is easy to check on the presence of multicolinearity amongst some variables. An example would be the readius_mean that has a corrleation of .99 and 11 with area_mean and perimeter_mean.

This does makes sense, because those three features basically are about the size of the tumor, so in some way or another, they have the same information. Another set of variables with multicolinearity are the ones of the 'worst' and 'mean' columns, ie.: area_worst and area_mean have a .96 correlation.

3. Create a data partition for Train and Test with their respective label. (Remember that your partition must be reproducible, that is, if you execute this instruction "n" times it must return the same partitions). *If your database already has an established partition, this is not necessary.
"""

#Drop the diagnosis column and asign it to y variable, converting it into our target
X=df.drop(['diagnosis'],axis=1)
y = df['diagnosis']

#Needed for the data partitition
from sklearn.model_selection import train_test_split

#code used to split the dataset into training set (X_train, y_train) and testing set (X_test,y_test)
#the partition is 30% for testing and 70% for training based on the protocol of Spanhol et al.
#that suggest a 70-30 partition in their Breast Cancer Histopathological Image Classification project.
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

"""
4. Create a pipeline that performs the data transformation.
"""

#Needed for the creation of the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler #Needed for the scaling of the data
import joblib  # Imports joblib directly

# Create  pipeline with the transformation of the data through scaling
pipeline = Pipeline([
    ('scaler', StandardScaler())])

#Adjust the  pipeline to the training and testing variables data
X_train = pipeline.fit_transform(X_train)
X_test= pipeline.fit_transform(X_test)

#Show first ten values of scaled training data
X_train[:10]

"""5. Save the data transformation (pipeline) in a .sav file."""

#This code is to save the pipeline into a file in drive
joblib.dump(pipeline, '/content/drive/MyDrive/FINAL_ML/pipeline_transformer.sav')

"""

6. Using a GridSearch or RandomizedSearchCV search for the best parameters for your LR model."""

#Needed for the ML model and the metrics that are gonna be used
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#Logistic Regression model
LR = LogisticRegression()

#Train the LR model on the training data
model_LR= LR.fit(X_train, y_train)

#Uses the trained model (model_LR) ro make predictions on the test dataset, and the values are stored in y_pred
y_pred = model_LR.predict(X_test)

#calculates the accuracy of the model comparing the value predicted and the actual value on the test set
print(metrics.accuracy_score(y_test, y_pred))

#SVM model
svm = SVC(kernel='rbf', random_state=42, gamma=.10, C=1.0)

#Train the SVM model on the training data
model_SVM= svm.fit(X_train, y_train)

#Uses the trained model (model_SVM) ro make predictions on the test dataset, and the values are stored in y_pred
y_pred = model_SVM.predict(X_test)

#calculates the accuracy of the model comparing the value predicted and the actual value on the test set
print(metrics.accuracy_score(y_test, y_pred))

#DecisionTree Model
DT = DecisionTreeClassifier(random_state=42)

#Train the DT model on the training data
model_DT= DT.fit(X_train, y_train)

#Uses the trained model (model_DT) ro make predictions on the test dataset, and the values are stored in y_pred
y_pred = model_DT.predict(X_test)

#calculates the accuracy of the model comparing the value predicted and the actual value on the test set
print(metrics.accuracy_score(y_test, y_pred))

#Dictionary created to have the lists of possible values for the gridsearch
LR = LogisticRegression()
SVM = SVC()
DT = DecisionTreeClassifier()

data = [(LR, [{'C': [0.01, 0.1, 0.5, 1.0], 'random_state':[42]}]),
        (SVM, [{'C': [0.1, 0.5, 1.0], 'kernel': ['linear', 'rbf'], 'random_state':[42]}]),
        (DT, [{'criterion':['gini', 'entropy'], 'max_depth':[None, 10, 20, 30, 40, 50], 'min_samples_split':[2, 5],'min_samples_leaf': [1, 2, 4]}])]

#Needed for GridSearch
from sklearn.model_selection import GridSearchCV

#Performing the gridsearch for hyperparameter tuning
#with their corresponding parameter grids (param_grid)
#Each combination of hyperparameters is evaluated with cross-validation of 10 folds,
#the best hyperparameters are retained in the grid object.

for i,j in data:
  grid = GridSearchCV(estimator = i , param_grid = j , scoring = 'accuracy',cv = 10)
  grid.fit(X_train,y_train) #Fits the gridsearch to the training data

  #Save best score of the grid, and the parameters it used
  best_accuracy = grid.best_score_
  best_parameters = grid.best_params_

  #Print best accuracy of the grid, and the best parameters
  print('{} \nBestAccuracy : {:.2f}%'.format(i,best_accuracy*100))
  print('BestParameters : ',best_parameters)

"""
7. Once you have identified the best parameters, use these to make your predictions with the Test data. If your models were trained with transformed data (e.g. standard scale), your test data set will need to be transformed."""

#Use LogisticRegression with best parameters obtained prior
best_LR = LogisticRegression(C=0.1, random_state=42)
#Fit the training data with the LR model
best_LR.fit(X_train, y_train)
#Use trained model to make predictions on test data
y_pred_LR = best_LR.predict(X_test)

#Use Support Vector Machine with best parameters obtained prior
best_SVM= SVC(C=0.1, kernel='linear', random_state=42)
#Fit the training data with the SVM model
best_SVM.fit(X_train, y_train)
#Use trained model to make predictions on test data
y_pred_SVM = best_SVM.predict(X_test)

#Use DecisionTree with best parameters obtained prior
best_DT = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=4, min_samples_leaf=5)
#Fit the training data with the DT model
best_DT.fit(X_train, y_train)
#Use trained model to make predictions on test data
y_pred_DT = best_DT.predict(X_test)

"""
8. Get your precision, recall and precision, F1 score, confusion matrix and classification report (classification_report)."""

#Needed for the metrics that are going to be measured
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

#Evaluate Logistic Regression with each king of metric, using the comparison of the prediction and the actual value of the test set
accuracy_LR = accuracy_score(y_test, y_pred_LR)
recall_LR = recall_score(y_test, y_pred_LR, average='weighted')
precision_LR = precision_score(y_test, y_pred_LR, average='weighted')
f1_LR = f1_score(y_test, y_pred_LR, average='weighted')
classification_report_LR = classification_report(y_test, y_pred_LR)

#Print results
print("Logistic Regression:")
print("Accuracy:", accuracy_LR)
print("Recall:", recall_LR)
print("Precision:", precision_LR)
print("F1 Score:", f1_LR)
print("Classification Report:")
print(classification_report_LR)

#Create and show a confusion matrix with the prediction and the actual value of the test set
confusion_matrix_LR = confusion_matrix(y_test, y_pred_LR)

confusion_matrix_LR

#Evaluate Support Vector Machine with each king of metric, using the comparison of the prediction and the actual value of the test set
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
recall_SVM = recall_score(y_test, y_pred_SVM, average='weighted')
precision_SVM = precision_score(y_test, y_pred_SVM, average='weighted')
f1_SVM = f1_score(y_test, y_pred_SVM, average='weighted')
classification_report_SVM = classification_report(y_test, y_pred_SVM)

#Print results
print("\nSVM:")
print("Accuracy:", accuracy_SVM)
print("Recall:", recall_SVM)
print("Precision:", precision_SVM)
print("F1 Score:", f1_SVM)
print("Classification Report:")
print(classification_report_SVM)

#Create and show a confusion matrix with the prediction and the actual value of the test set
confusion_matrix_SVM = confusion_matrix(y_test, y_pred_SVM)

confusion_matrix_SVM

#Evaluate Decision Tree with each king of metric, using the comparison of the prediction and the actual value of the test set
accuracy_DT = accuracy_score(y_test, y_pred_DT)
recall_DT = recall_score(y_test, y_pred_DT, average='weighted')
precision_DT = precision_score(y_test, y_pred_DT, average='weighted')
f1_DT = f1_score(y_test, y_pred_DT, average='weighted')
classification_report_DT = classification_report(y_test, y_pred_DT)


#Print results
print("\nDecision Tree:")
print("Accuracy:", accuracy_DT)
print("Recall:", recall_DT)
print("Precision:", precision_DT)
print("F1 Score:", f1_DT)
print("Classification Report:")
print(classification_report_DT)

#Create and show a confusion matrix with the prediction and the actual value of the test set
confusion_matrix_DT = confusion_matrix(y_test, y_pred_DT)

confusion_matrix_DT

"""9. Store your trained model in a .sav file.


"""

#Assign the path of where the files will be stored
LR_model_file = '/content/drive/MyDrive/FINAL_ML/LR_model.sav'
SVM_model_file = '/content/drive/MyDrive/FINAL_ML/SVM_model.sav'
DT_model_file = '/content/drive/MyDrive/FINAL_ML/DT_model.sav'


#Save the trained models as a .sav files
joblib.dump(best_LR, LR_model_file)
joblib.dump(best_SVM, SVM_model_file)
joblib.dump(best_DT, DT_model_file)

#Libraries needed for ensamble
from sklearn.ensemble import VotingClassifier

#Create the ensamble of voting classifier
VC = VotingClassifier(
    #The estimators argument is a list of tuples where each tuple contains the name of the base classifier and the base classifier itself
    estimators=[("Logistic Regression",best_LR), ("Support Vector Machine",best_SVM), ("Decision Tree",best_DT)],
    #The argument voting='hard' indicates that the majority vote will be used for the prediction.
    voting='hard')

#Create a loop that iterates over a list containing the base classifiers and the voting classifier
for clf, label in zip([best_LR, best_SVM, best_DT, VC], ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Voting Classifier']):
    #For each classifier, cross validation is calculated using 5 partitions and measuring accuracy
    scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=5)
    #Show the mean precision and standard deviation of the cross-validation scores for each classifier
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

best_VC = VC
#Fit the training data with the VC model
best_VC.fit(X_train, y_train)
#Use trained model to make predictions on test data
y_pred_VC = best_VC.predict(X_test)

#Evaluate Ensemble with each king of metric, using the comparison of the prediction and the actual value of the test set
accuracy_VC = accuracy_score(y_test, y_pred_VC)
recall_VC = recall_score(y_test, y_pred_VC, average='weighted')
precision_VC = precision_score(y_test, y_pred_VC, average='weighted')
f1_VC = f1_score(y_test, y_pred_VC, average='weighted')
classification_report_VC = classification_report(y_test, y_pred_VC)


#Print results
print("\Ensemble:")
print("Accuracy:", accuracy_VC)
print("Recall:", recall_VC)
print("Precision:", precision_VC)
print("F1 Score:", f1_VC)
print("Classification Report:")
print(classification_report_VC)

#Assign the path of where the file will be stored, and save it as a .sav file
VC_model_file = '/content/drive/MyDrive/FINAL_ML/VC_model.sav'

joblib.dump(VC, VC_model_file)



"""**References:**

Spanhol, F.A., Oliveira, L.S., Petitjean, C., Heutte, L.(2016). A dataset for breast cancer histopathological image classification. IEEE Trans. Biomed. Eng. 63(7), 1455–1462. https://doi.org/10.1109/TBME.2015.2496264

"""