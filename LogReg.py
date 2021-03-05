
# coding: utf-8

# In[52]:


# We need to ensure we set the correct environement for predictive model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Confusion matrix import
from sklearn.metrics import confusion_matrix
# Splitting data
from sklearn.model_selection import train_test_split
# model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# iloc is a panda dataframe, here we are selectin all the rows and the last column of the data
X = HeartData.iloc[:, :-1]
y = HeartData.iloc[:, -1]

# Here we simply assinging the data traing and test chunk to the x and y axis, we split the data 75% train and 
# therefore 25% test size 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[53]:


# Assignining the 'LogisticRegression' model from the sklearn library 
LogisticR = LogisticRegression()

# using the fit function to take the first paremter to the X axis and second paramter as the y axis which the fit 
# function requires, we training the data here
LogisticR.fit(X_train, y_train)

#prediction
y_pred = LogisticR.predict(X_test)

# This is will represent the logisitic regression 
print("Accuracy ", LogisticR.score(X_test, y_test)*100)

# We use the confusion matrix to confirm the model accruacy determining how instances the model predicted correctly 
# and incorrectly,
# you can see that the model correctly classified 82.9% of the instances of hevaing heart disease, this would indicate 
# a good predictive model
sns.set(font_scale=2.0)
cm = confusion_matrix(y_pred, y_test)
# you pass in th matrix, ensure it will be annotated and also the format type
sns.heatmap(cm, annot=True, fmt='g')
plt.show()


# In[54]:


# This is accuracy of the the decision tree from the random forest classifier----------------------
from sklearn.ensemble import RandomForestClassifier #for the model

#Model
deciTree = RandomForestClassifier()

#fiting the model
deciTree.fit(X_train, y_train)

#prediction
y_pred = deciTree.predict(X_test)

#Accuracy
print("Accuracy ", deciTree.score(X_test, y_test)*100)

#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()
# -------------------------------------------------------------------


# In[55]:


from sklearn.metrics import roc_curve, auc #for model evaluation
y_pred_quant = LogisticR.predict_proba(X_test)[:, 1]
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


# In[56]:


auc(fpr, tpr)

