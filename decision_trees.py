## Decision Trees
## Author: Konstantinos Gyftodimos

# Importing Libraries.
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
import pandas as pd
import plotly.express as px
from pydotplus import graph_from_dot_data
from IPython.display import Image

# Turn off Numpy showing numbers in a scientific notation.
np.set_printoptions(suppress=True)

# Load dataset
df = pd.read_csv('datasets/data_decision_trees.csv')
print(df)

# Visualize dataset
data_dimensions = df.columns[:-1].to_list()
figure_size = df.shape[1] * 256
fig = px.scatter_matrix(df, dimensions=data_dimensions, color='Label', width=figure_size, height=figure_size)
fig.show()

# Turn dataset to Numpy Arrays to put it after into sklearn algorithms.
X = df.drop('Label', axis=1).to_numpy()
y = df['Label'].to_numpy()
print('Shape of X array is:', X.shape)
print('Shape of y array is:', y.shape)

# Splitting Data. Sklearn cannot split the dataset into 3 subsets immediatelly. 
# So first i split all data to Training and Validation/Testing.
# Then i split the second subset to Validation and Testing.
(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size=0.4, random_state=0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)

# print('Shape of Train/Validation/Test data:  ',X_train.shape, y_train.shape, 
# 											   X_validation.shape, y_validation.shape, 
# 											   X_test.shape, y_test.shape)

# Build Decision Tree Classifier 
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

# Visualization of Decision Tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, impurity=False, special_characters=True)
graph = graph_from_dot_data(dot_data.getvalue())

# Model Assessment and selection
yhat_train=dtree.predict(X_train)
print(accuracy_score(yhat_train, y_train)) 
# We see a perfect 100% because minimum leaf size is 1.
# We have Complete Seperation here.
# So we have overfitting
# I bound the minimum samples per leaf instead of 1 to 15, by creating another tree:
# I could also upper bound the max num of leaves + other techniques.
dtree2 = DecisionTreeClassifier(min_samples_leaf = 15)
dtree2 = dtree2.fit(X_train,y_train)

# Fit the data - Training
yhat_train2=dtree2.predict(X_train)
print('Accuracy on training data: ',accuracy_score(yhat_train2, y_train))
# Validation 
yhat_validation2=dtree2.predict(X_validation)
print('Accuracy on validation data: ',accuracy_score(yhat_validation2, y_validation))
# Testing
yhat_test2=dtree2.predict(X_test)
print('Accuracy on test data: ',accuracy_score(yhat_test2, y_test))








