## Support Vector Machines, testing on different Kernels for binary classification
## Author: Konstantinos Gyftodimos

# Importing Required Libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.graph_objs as go

# Loading the data and printing the dataframe:
df = pd.read_csv('datasets/data_svms_and_kernels.csv')
print(df)

# Splitting Features and Classes
X = df.drop('Label', axis=1).to_numpy()
print(X.shape)

y = df['Label'].to_numpy()
print(y.shape)

# Split the data:
(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size=0.4, random_state=0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)

# Build an SVM with a Linear Kernel:
svm = SVC(kernel='linear')
svm = svm.fit(X_train,y_train)

# Build an SVM with a polynomial Kernel of p=2:
svm_p2 = SVC(kernel='poly',degree=2)
svm_p2 = svm_p2.fit(X_train,y_train)

# Build an SVM with a polynomial Kernel of p=3:
svm_p3 = SVC(kernel='poly',degree=3)
svm_p3 = svm_p3.fit(X_train,y_train)

# Build an SVM with a RBF Kernel:
svm_r = SVC(kernel = 'rbf')
svm_r = svm_r.fit(X_train,y_train)

# Model Selection Phase, based on Training and Validation data:

#Linear SVM
print('Linear SVM performance on train and validation data:')
yhat_train = svm.predict(X_train)
yhat_validation = svm.predict(X_validation)
print(accuracy_score(yhat_train, y_train),accuracy_score(yhat_validation, y_validation))
#Polynomial SVM, p=2
print('Polynomial SVM, p=2, performance on train and validation data:')
yhat_train_p2 = svm_p2.predict(X_train)
yhat_validation_p2 = svm_p2.predict(X_validation)
print(accuracy_score(yhat_train_p2, y_train), accuracy_score(yhat_validation_p2, y_validation))
#Polynomial SVM, p=3
print('Polynomial SVM, p=3, performance on train and validation data:')
yhat_train_p3 = svm_p3.predict(X_train)
yhat_validation_p3 = svm_p3.predict(X_validation)
print(accuracy_score(yhat_train_p3, y_train), accuracy_score(yhat_validation_p3, y_validation))
# RBF - SVM
print('RBF SVM performance on train and validation data:')
yhat_train_r = svm_r.predict(X_train)
yhat_validation_r = svm_r.predict(X_validation)
print(accuracy_score(yhat_train_r, y_train), accuracy_score(yhat_validation_r, y_validation))


# I trust more the RBF model, now lets see how it works on Test data:
print('RBF SVM performance on test and data:')
yhat_test_r = svm_r.predict(X_test)
print(accuracy_score(yhat_test_r, y_test))

# Last comments: I see that data are easilly seperable so a linear kernel would be enough.
# Nevertheless, i wanted to see how it operates with other kernels. 







