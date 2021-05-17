## Logistic Regression, using PCA to reduce dimensionality.
## Author: Konstantinos Gyftodimos

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import csv

## Loading the Dataset and previewing the current data frame:
df_30 = pd.read_csv('datasets/data_logistic_regression.csv')
print(df_30)

## Seperate the data:
X_30 = df_30.drop('type', axis=1).to_numpy()
print('30D-dimensional data shape: ',X_30.shape)
y_text = df_30['type'].to_numpy()
print('output binary labels: ',y_text.shape)

## For every class output we have 30-features. I will reduce the dimensionality from 30
## to 2-dimensions, using Principal Component Analysis
pca = PCA(n_components=2)
pca.fit(X_30)
X = pca.transform(X_30)
print('2D-dimensional data shape, after PCA: ',X.shape)

## Generate the new dataframe and visualize it:
df = pd.DataFrame(data=np.c_[X, y_text], columns=['Feature 1', 'Feature 2', 'Label'])
fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Label')
fig.show()

## Tranform the categorical labels to -1 and +1
y = (2 * LabelEncoder().fit_transform(y_text)) - 1

## Plot in 3D the two features and the corresponding label:
points_colorscale = [
                     [0.0, 'rgb(239, 85, 59)'],
                     [1.0, 'rgb(99, 110, 250)'],
                    ]

layout = go.Layout(scene=dict(
                              xaxis=dict(title='Feature 1'),
                              yaxis=dict(title='Featrue 2'),
                              zaxis=dict(title='Label')
                             ),
                  )

points = go.Scatter3d(x=df['Feature 1'], 
                      y=df['Feature 2'], 
                      z=y,
                      mode='markers',
                      text=df['Label'],
                      marker=dict(
                                  size=3,
                                  color=y,
                                  colorscale=points_colorscale
                            ),
                     )

fig2 = go.Figure(data=[points], layout=layout)
fig2.show()


# Split the PCA-reduced data into train and test ratios (70/30%)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)

# Build and Visualize Logistic Regression Model:
classifier = LogisticRegression()
classifier = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_pred, y_test)
print('Total Accuracy Score: ',acc)














