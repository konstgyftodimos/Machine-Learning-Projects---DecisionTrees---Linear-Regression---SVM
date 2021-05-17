# Machine-Learning-Projects-DecisionTrees-Linear-Regression-SVM

In the current project i apply a number of ML-algorithms, while giving attention to data spreadability and seperability.

# SVM - linear - polynomial (p=2,3) - RBF

After spitting the features and pre-processing the data i visualized the data and applied the following kernels to observe their seperability and classification rate:
1. linear 
2. polynomial of 2nd order
3. polynomial of 3rd order
4. RBF

# Logistic Regression 

Before applying Logistic Regression i need to seperate my data. Due to the high number of features in the dataset i reduced the dimensionality of highly correlated features with PCA.

Data before PCA (30 features):

![](images/logr_data.png | width="200" height="200")

Data after PCA (2 features):

![](images/logr_data_after_pca.png =100x20)

# Decision Trees

Build a decision tree classifier, with minimum_sample_leaves = 15 to avoid overfitting. 
An overview of what data looks like can be described by the scatter matrix below:

![](images/scatterplot_dt.png =100x20)

