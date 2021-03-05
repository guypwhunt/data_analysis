
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings


# In[53]:


# Import heart data assign it to HeartData variable
HeartData = pd.read_csv('data/cleansed-healthcare-dataset-stroke-data.csv',delimiter=',',header='infer')
print(HeartData.columns)

# In[55]:


HeartData = pd.get_dummies(HeartData, drop_first=True)
print(HeartData)


# In[56]:


print(HeartData.head())


# In[57]:


# this splits the 08
X_train, X_test, y_train, y_test = train_test_split(HeartData.drop('stroke', 1), HeartData['stroke'], test_size = .2, random_state=10) #split the data


# In[115]:


# Give the depth paramter of the to the decsion tree to stop the tree after level of seperation in which each tree 
# will divide the parameters. 
model = RandomForestClassifier(max_depth=5)
# this will fit the training data to the tree 
model.fit(X_train, y_train)
# model.fit(X_test, y_test) #this will plot the tree for the test data 


# In[116]:


estimator = model.estimators_[1]
# These are the column names in the dataset, so each node in the tree can be named
# you loop through the column names and plave them in a variable for later use
features = [i for i in X_train.columns]

# You convert the training data to type string so the output can be classified as the desired output, you can see
# in the following lines
y_train_str = y_train.astype('str')
# Here it will interpert the reuslt at the end of the tree being either 0 or 1 and converting them to no disease or 
# making it easier to understand for the user
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
# Then you convert them baack to values so the export graph viz function can use them. 
y_train_str = y_train_str.values


# In[121]:


# The the eport graphviz function will take in the paramters you apply output a file in the case a dot file
# you can specify the type of file you would specifically like 
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = features,
                class_names = y_train_str, 
                proportion = True, 
                label='root',
                precision = 2, filled = True)

# this Process will conver the .dot file into a png so it can be view
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'DecisionTree.png', '-Gdpi=600'])

# Here is allows it to be saved the local directory
from IPython.display import Image
Image(filename = 'DecisionTree.png')
# The result will be a tree with the training data


# In[118]:


# This is just a alternate way of viewing the deicsion, we use the same library and function be do not define the
# paremters with the samw detail and output reflects this
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(estimator, out_file=None)
graphviz.Source(tree_graph)


# In[119]:


# This is the values for the confusion matirx, this the test and prediction comparison placed into the 
# cunfusion matrix 
y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_bin)
cm


# In[126]:


total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', specificity)


# In[123]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 10
plt.title('ROC curve Heart disease')
plt.xlabel('False Positive Rate FPR')
plt.ylabel('True Positive Rate TPR')
plt.grid(True)


# In[125]:


auc(fpr, tpr)

