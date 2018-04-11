#Importing the required models and libraries.

#To run this file one should have Xgboost,scikitlearn and the corresponding libraries installed.

import pandas                                          #Data analysis tools for the Python.
import matplotlib.pyplot as plt                        #Plotting tools for Python.
from sklearn import model_selection                    #Sklearn or scikit-learn is a machine learning library.
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor     #Importing different models.
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor                        #Xgboost is also a machine learning model.
                                                        #XGBRegressor is for regression models.

import sys
from sklearn.metrics import mean_squared_error,mean_absolute_error #Different error calculation methods.
import time

url= sys.argv[1]                # Storing the value received as argument, during the file call, python model.py.
                                #It should be the name of the file, keeping dataset in CSV format.
dataset = pandas.read_csv(url)  #Reading the contents of the file and storing it to the DataFrame.
array = dataset.values          #Dataframe is represented as an array.
X = array[:,1:13]               #Slicing the input parameters .i.e from columns B to M and storing it to array X.
Y = array[:,0]                  #Storing the output values to array Y.

validation_size = 0.23
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#train_test_split: Randomly splits 23% rows of data and is kept in X_validation and Y_validation.
#The remaining are kept in X_train and Y_train.
#This helps to avoid overfitting
#The data in X_validation and Y_validation are never used during training.This data is used to
#determine the accuracy of the model after training
#test_size :represent the proportion of the dataset to be included in the test split.
#random_state is the seed used by the random number generator

seed = 7
models = []

#evaluate each model in turn
models.append(('SVM', SVR()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('LinearRegression', LinearRegression()))      #Initializing different regression models ,for evaluation of the fit
models.append(('CNN', MLPRegressor()))
models.append(('XGB', XGBRegressor()))
results = []
names = []
print"      Executing cross-validation on 77% data...."
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold) #RMS
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f " % (name, cv_results.mean())
	print(msg)
#The quality of the fit will be assessed using cross-validation with RMS
#The trainingdata is split into groups of 10 rows ,where in each split 9 rows are used for training and  10th row for testing



print"\n     Selecting the XGBRegressor AS it it has maximum accuracy.....\n"
#Creating a new XGBRegressor with different parameters tuned to obtain maximum accuracy
xgb = XGBRegressor(
                    max_depth=5, # Maximum tree depth for base learners
                    learning_rate= 0.065,#Boosting learning rate
                    n_estimators=500,#Number of boosted trees to fit
                     )
xgb.fit(X_train, Y_train)   #Training the regressor model on the 77% trainingdata
predictionsv = xgb.predict(X_validation) #Using the trained model to predict on a new dataset (X_validation),that the model has never seen


fullpredict =xgb.predict(X)  #Predicting on the input parameters of each row of the dataset
scoreonvalidation=xgb.score(X_validation, Y_validation)#Score is the accuracy.This is the accuracy of the model on validation set

print "**Details of prediction on the (23%) validation set**"
print"score                   :",scoreonvalidation
print"Root Mean Square Error  :",mean_squared_error(predictionsv, Y_validation)**0.5#root mean squared error
print"Average Absolute Error  :",mean_absolute_error(Y_validation,predictionsv)      #average absolute error
error= fullpredict - Y                               #error on each row of predictions on the whole dataset
errorv = predictionsv - Y_validation                 #error on each row of predictions on the validation set
print"Max Absolute Error      :",max(error**2)**0.5                               #max absolute error

print"\n**Details of prediction on the whole dataset**"
print"score                   :",xgb.score(X, Y)           #accuracy of predictions on the whole dataset of the CSV file
print"Root Mean Square Error  :",mean_squared_error(fullpredict, Y)**0.5#root mean squared error
print"Average Absolute Error  :",mean_absolute_error(Y,fullpredict)#Average absolute error
print"Max Absolute Error      :",max(errorv**2)**0.5               #Maximum absolute error
plt.figure(figsize=(19,9))                                             #setting the size of the window to plot(in inches)
pred, = plt.plot(fullpredict,'.')

true,=plt.plot(Y,'r')
plt.legend([pred,true], ["Predicted Values","True Values"])
plt.title("Error Plot")
plt.ylabel("Values")
plt.xlabel("Rows")
#plt.plot(error)                                                #plotting the error for predictions on the whole datset
#plt.title('Plot of errors for each row')
#plt.ylabel('Error(Predicted value-Actual value)')
#plt.xlabel('Rows')
plt.show()
