# DA_Project
# Importing the libraries we need
```python
#importing the libraries that we use in our project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn import linear_model, metrics

```
# Creating the DataFrames
```python
kill= pd.read_csv("F:\Les\Data analytic\Project\Data/PoliceKillingsUS.csv", encoding='latin-1')
poverty_level = pd.read_csv("F:\Les\Data analytic\Project\Data/PercentagePeopleBelowPovertyLevel.csv",encoding='latin-1')
median_house_income = pd.read_csv("F:\Les\Data analytic\Project\Data/MedianHouseholdIncome2015.csv",encoding='latin-1')
race_by_city = pd.read_csv("F:\Les\Data analytic\Project\Data/ShareRaceByCity.csv",encoding='latin-1')
percentage_complete_highschool = pd.read_csv("F:\Les\Data analytic\Project\Data/PercentOver25CompletedHighSchool.csv",encoding='latin-1')
```
#  The structure of the police killings DataFrames
```python
kill.head(10)
```
#  The structure of the 'poverty_level' DataFrame
```python
poverty_level.head(10)
```
#  The structure of the 'median_house_income' DataFrame
```python
median_house_income.head(10)
```
#  The structure of the 'race_by_city' DataFrame
```python
race_by_city.head(10)
```
# The structure of the 'percentage_complete_highschool' DataFrame
```python
percentage_complete_highschool.head(10)
```
# Cleaning US Police shooting (kill DataFrame)
```python
kill.drop_duplicates() #removing duplicates
kill_miss_values = kill.isnull().sum() #see the kill DataFrame attributes that have Nan values
kill.armed = kill.armed.fillna('unknown') #filling the armed Nan values by 'unknown'
kill['age'].fillna((kill['age'].mean()), inplace=True) #filling the age Nan values by the average 
kill.dropna(subset=['race'],how='all', inplace = True) #drop the rows that have Nan values in 'race' attribute
kill.drop('flee',axis=1,inplace=True) #droping the 'flee' attribute
kill['name'].replace(['TK TK'],'Unknown',inplace = True) #Replacing the rows that have 'TK TK' in 'name' attribute bt 'Unknown'
kill.armed = kill.armed.str.lower() #making the armed column to lower
kill.armed = kill.armed.mask(kill.armed == 'undetermined','unknown') #replacing variations with uniform values
kill.armed = kill.armed.mask(kill.armed == 'unknown weapon','unknown')
kill.armed= kill.armed.mask(kill.armed == 'guns and explosives','gun')
kill.armed= kill.armed.mask(kill.armed == 'gun and knife','gun') 
kill.armed= kill.armed.mask(kill.armed == 'hatchet and gun','gun')
kill.armed= kill.armed.mask(kill.armed == 'machete and gun','gun')
kill.armed = kill.armed.mask(kill.armed == 'sword','knife')
kill.armed = kill.armed.mask(kill.armed == 'machete','knife')
kill.armed = kill.armed.mask(kill.armed == 'box cutter','knife')
kill.armed = kill.armed.mask(kill.armed == 'bayonet','knife')
kill.armed = kill.armed.mask(kill.armed == 'meat cleaver','knife')
kill.armed = kill.armed.mask(kill.armed == 'pole and knife','knife')
kill.armed = kill.armed.mask(kill.armed == 'lawn mower blade','knife')
kill.armed = kill.armed.mask(kill.armed == 'straight edge razor','knife')
kill.armed = kill.armed.mask(kill.armed == 'motorcycle','vehicle')
kill.armed = kill.armed.mask (kill.armed == 'ax','hatchet')
kill.armed = kill.armed.mask(kill.armed == 'flagpole','metal object')
kill.armed = kill.armed.mask(kill.armed == 'metal pipe','metal object')
kill.armed = kill.armed.mask(kill.armed == 'metal hand tool','metal object')
kill.armed = kill.armed.mask(kill.armed == 'metal pole','metal object')
kill.armed = kill.armed.mask(kill.armed == 'metal stick','metal object')
kill.armed = kill.armed.mask(kill.armed == 'brick','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'baseball bat','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'pole','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'rock','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'piece of wood','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'baton','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'oar','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'baseball bat and fireplace poker','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'hammer','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'baseball bat and bottle','blunt object')
kill.armed = kill.armed.mask(kill.armed == 'pipe','blunt object')
kill.drop('id',axis=1,inplace=True) #dropping the id attribute
kill.columns =[ 'name','date','manner_of_death','armed','age','gender','race','city','state','mental_illness','threat',
               'body_camera'] #renaming the kill DataFrame column
kill.city = kill.city.str.lower() #making city to lowercase
kill.date = pd.to_datetime(kill.date,dayfirst=True) #ensure that date column is of date type
```
# Police killings after cleaning

# Cleaning poverty level
```python
poverty_level.drop_duplicates() #removing the duplicates
poverty_level_miss_values = poverty_level.isnull().sum()#see the poverty_level DataFrame attributes that have Nan values
poverty_level = poverty_level [poverty_level.iloc[:,2] != '-']#removing rows that have '-' in poverty_rate
poverty_level.columns= ['state','city','poverty_rate']#renaming columns 
poverty_level.city = poverty_level.city.str.lower()
poverty_level.poverty_rate = pd.to_numeric(poverty_level.poverty_rate,downcast='float')#ensure that poverty rate column is of float
```
# poverty level after cleaning

# Cleaning median house income
```python
median_house_income.drop_duplicates() #removing duplicates
#see the Nan values of the median_house_income DataFrame
median_house_income_miss_values = median_house_income.isnull().sum()
#dropping the rows that have Nan values in median income
median_house_income.dropna(subset=['Median Income'],how='all', inplace = True)
#drop (X) and '-' values
median_house_income = median_house_income [median_house_income.iloc[:,2] != '(X)'] 
median_house_income = median_house_income [median_house_income.iloc[:,2] != '-']
#replacing median income which is -2500 with 2500
median_house_income.iloc[:,2] = median_house_income.iloc[:,2].replace('2,500-',2500) 
median_house_income.iloc[:,2] = median_house_income.iloc[:,2].replace('250,000+',250000)
#Rename columns
median_house_income.columns = ['state','city','median_income'] 
#make the city column to lower
median_house_income.city = median_house_income.city.str.lower()
median_house_income.median_income = pd.to_numeric(median_house_income.median_income,downcast='float')
```
# Cleaning race by city
```python
#removing duplicates
race_by_city.drop_duplicates()
#see the Nan values of the race_by_city DataFrame
race_by_city_miss_values = race_by_city.isnull().sum()
#replacing (X) values in race columns
race_by_city = race_by_city [race_by_city.iloc[:,2] != '(X)']
race_by_city = race_by_city [race_by_city.iloc[:,3] != '(X)']
race_by_city = race_by_city [race_by_city.iloc[:,4] != '(X)']
race_by_city = race_by_city [race_by_city.iloc[:,5] != '(X)']
race_by_city = race_by_city [race_by_city.iloc[:,6] != '(X)']
#replacing - values in race columns
race_by_city = race_by_city [race_by_city.iloc[:,2] != '-']
race_by_city = race_by_city [race_by_city.iloc[:,3] != '-']
race_by_city = race_by_city [race_by_city.iloc[:,4] != '-']
race_by_city = race_by_city [race_by_city.iloc[:,5] != '-']
race_by_city = race_by_city [race_by_city.iloc[:,6] != '-']
#renaming columns
race_by_city.columns= ['state','city','white','black','native_american','asian','hispanic']
race_by_city.city = race_by_city.city.str.lower()
#ensure that race percentage column is of float
race_by_city.white = pd.to_numeric(race_by_city.white, downcast='float')
race_by_city.black = pd.to_numeric(race_by_city.black, downcast='float')
race_by_city.native_american = pd.to_numeric(race_by_city.native_american, downcast='float')
race_by_city.asian = pd.to_numeric(race_by_city.asian, downcast='float')
race_by_city.hispanic = pd.to_numeric(race_by_city.hispanic, downcast='float')
```
# Cleaning complete highschool 
```python
#removing duplicates
percentage_complete_highschool.drop_duplicates()
#see the Nan values of the percentage_complete_highschool DataFrame
percentage_complete_highschool_miss_values = percentage_complete_highschool.isnull().sum()
#dropping the rows that have - in their rate
percentage_complete_highschool = percentage_complete_highschool [percentage_complete_highschool.iloc[:,2] != '-']
#renaming columns
percentage_complete_highschool.columns = ['state','city','completed_hs']
percentage_complete_highschool.city = percentage_complete_highschool.city.str.lower()
percentage_complete_highschool.completed_hs = pd.to_numeric(percentage_complete_highschool.completed_hs, downcast='float')
```
# People killed by race 
```python
#showing the total number of people killed by race
plt.figure(figsize=(15,5))
sns.countplot(data=kill, x='race')
plt.title('Total number of people killed, by race',fontsize=18)
```
# People killed as a proportion of their race
```python
# showing the number of people killed as a proportion of their respective race
race_list = ["A", "B", "H", "N", "O", "W"]
killed_perrace = []

for i in race_list:
    killings_c = kill.race.loc[(kill.race==i)].count()
    killed_perrace.append(killings_c)
    
print (killed_perrace) #showing killed people for each races
proportion_killed_perrace = []

for i in race_list:
    if i == "A":
        proportion_killings_c = killed_perrace[0]/14674252.0
        print (proportion_killings_c)
    elif i == "B":
        proportion_killings_c = killed_perrace[1]/38929319.0
        print (proportion_killings_c)
    elif i == "H":
        proportion_killings_c = killed_perrace[2]/50477594.0
        print (proportion_killings_c)
    elif i == "N":
        proportion_killings_c = killed_perrace[3]/2932248.0
        print (proportion_killings_c)
    elif i == "O":
        proportion_killings_c = killed_perrace[4]/22579629.0
        print (proportion_killings_c)
    else:
        proportion_killings_c = killed_perrace[5]/223553265.0
        print (proportion_killings_c)
        
    proportion_killed_perrace.append(proportion_killings_c)
plt.figure(figsize=(14,6))
sns.barplot(x=race_list, y=proportion_killed_perrace)        
```
# Number of People killed by state
```python
plt.figure(figsize=(20,10))
sns.countplot(data=kill, x=kill.state, order=kill['state'].value_counts().index)
plt.title("Number of police killings, by state", fontsize=27)
```

# Age distribution of people murdered by police
```python
# General age distribution
plt.figure(figsize=(15,5))
age_dist = sns.distplot(kill.age, bins= 40)
age_dist.set(xlabel='Age',ylabel='Count')
plt.title('Age distribution',fontsize=18)
```
# Age distribution of people murdered by race
```python
#age distribution by race
three_races = kill.loc[(kill["race"] == "B") | (kill["race"] == "W") | (kill["race"] == "H")]
g = sns.FacetGrid(data=three_races, hue="race", aspect=3, size=4)
g.map(sns.kdeplot, "age", shade=True)
g.add_legend(title="Race")
g.set_ylabels("Count")
plt.title("Age distribution, by race", fontsize=17)
```

# Age distribution of people murdered in the states with highest fatalities
```python
plt.figure(figsize=(20,10))
sns.boxplot(x='state', y='age',data=kill, order=['CA','TX','FL','AZ','OH','OK','NC','CO','GA','IL'])
```
# States with the highest poverty rate
```python
area_list = list(poverty_level['state'].unique())
poverty_ratio=[]
#calculate the eaverage rate for each city in terms of their cities
for i in area_list: 
    x=poverty_level[poverty_level['state']== i]
    area_poverty_rate= sum(x.poverty_rate)/len(x)
    poverty_ratio.append(area_poverty_rate)
data_poverty_level = pd.DataFrame({'area_list': area_list , 'poverty_rate':poverty_ratio})
poverty_level_index = (data_poverty_level['poverty_rate'].sort_values(ascending = False )).index.values
data_poverty_level = data_poverty_level.reindex(poverty_level_index)
data_poverty_level.head(10)
plt.figure(figsize=(15,10))
g=sns.barplot(x=area_list,y=data_poverty_level.poverty_rate,data=data_poverty_level)
g.set(xlim=(0, 19)) #to see the top20
```
# States with the lowest high school completion rate
```python
highschool_ratio=[]
#calculate highschool rate for each state
for i in area_list:
    x = percentage_complete_highschool[percentage_complete_highschool['state']==i]
    complete_rate = sum(x.completed_hs)/len(x)
    highschool_ratio.append(complete_rate)
HS_Ratio = pd.DataFrame({'area_list': area_list , 'HighSchool_Rate': highschool_ratio})
HS_Ratio_index = (HS_Ratio['HighSchool_Rate'].sort_values(ascending = True )).index.values
sorted_HS_Ratio = HS_Ratio.reindex(HS_Ratio_index)
plt.figure(figsize=(15,10))
g=sns.barplot(x=sorted_HS_Ratio.area_list,y=sorted_HS_Ratio.HighSchool_Rate,data=sorted_HS_Ratio)
g.set(xlim=(0, 19)) #getting the top20
```
# Correlation between poverty and highschool graduation rates
```python
area_list = list(poverty_level['state'].unique())
poverty_ratio=[]
for i in area_list:
    x=poverty_level[poverty_level['state']== i]
    area_poverty_rate= sum(x.poverty_rate)/len(x)
    poverty_ratio.append(area_poverty_rate)
data_poverty_ratio = pd.DataFrame({'area_list': area_list , 'poverty_rate':poverty_ratio})
poverty_ratio_index=(data_poverty_ratio['poverty_rate'].sort_values(ascending=False)).index.values
sorted_poverty_ratio = data_poverty_ratio.reindex(poverty_ratio_index)

highschool_ratio=[]
for i in area_list:
    x = percentage_complete_highschool[percentage_complete_highschool['state']==i]
    complete_rate = sum(x.completed_hs)/len(x)
    highschool_ratio.append(complete_rate)
HS_Ratio = pd.DataFrame({'area_list': area_list , 'HighSchool_Rate': highschool_ratio})
HS_Ratio_index = (HS_Ratio['HighSchool_Rate'].sort_values(ascending = False )).index.values
sorted_HS_Ratio = HS_Ratio.reindex(HS_Ratio_index)
data=pd.concat([sorted_poverty_ratio,sorted_HS_Ratio["HighSchool_Rate"]],axis=1)
g = sns.jointplot(data.poverty_rate, data.HighSchool_Rate, kind="reg", size=7)
```
# How the victims were armed
```python
armed=kill.armed.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(armed[:7].index,armed[:7].values)
```
# The mental situation
```python
plt.figure(figsize=(20,10))
sns.countplot(data=kill, x='mental_illness')
```
# visualise the race distribution in each state
```python
area_list = list(race_by_city['state'].unique())
white=[]
black=[]
native_american=[]
asian=[]
hispanic=[]
for i in area_list:
    x=race_by_city[race_by_city["state"]==i]
    white.append(sum(x.white)/len(x))
    black.append(sum(x.black)/len(x))
    native_american.append(sum(x.native_american)/len(x))
    asian.append(sum(x.asian)/len(x))
    hispanic.append(sum(x.hispanic)/len(x))
f,ax=plt.subplots(figsize=(15,10))
sns.barplot(x=white,y=area_list,color="#8c001a", alpha=0.9,label="Whites")
sns.barplot(x=black,y=area_list, color= "#00fdd1", alpha=0.9, label="Blacks")
sns.barplot(x=native_american,y=area_list, color= "#2701d5", alpha=0.9, label="Native Americans")
sns.barplot(x=asian,y=area_list, color="#ffd62a", alpha=0.9, label="Asians")
sns.barplot(x=hispanic,y=area_list, color="#46a346", alpha=0.9, label="Hispanics")

ax.legend(ncol=2,loc="upper right",frameon=True)    
```
# Modeling

# Removing prefix for each city name
```python
#removing 'cdp' and 'city' and 'town' and 'village' from the end of each city name
for index,row in poverty_level.iterrows():
    tempcity = row["city"]
    citysplit = tempcity.split(" ")
    if(citysplit[-1] == "cdp" or citysplit[-1] == "city" or  citysplit[-1] == "town" or citysplit[-1]=="village"): 
        del citysplit[-1] 
    tempstring = " ".join(citysplit)
    poverty_level.set_value(index,"city",tempstring)

    
for index,row in median_house_income.iterrows():
    tempcity = row["city"]
    citysplit = tempcity.split(" ")
    if(citysplit[-1] == "cdp" or citysplit[-1] == "city" or  citysplit[-1] == "town" or citysplit[-1]=="village"): 
        del citysplit[-1] 
    tempstring = " ".join(citysplit)
    median_house_income.set_value(index,"city",tempstring)
    
for index,row in percentage_complete_highschool.iterrows():
    tempcity = row["city"]
    citysplit = tempcity.split(" ")
    if(citysplit[-1] == "cdp" or citysplit[-1] == "city" or  citysplit[-1] == "town" or citysplit[-1]=="village"): 
        del citysplit[-1] 
    tempstring = " ".join(citysplit)
    percentage_complete_highschool.set_value(index,"city",tempstring)
    
for index,row in race_by_city.iterrows():
    tempcity = row["city"]
    citysplit = tempcity.split(" ")
    if(citysplit[-1] == "cdp" or citysplit[-1] == "city" or  citysplit[-1] == "town" or citysplit[-1]=="village"): 
        del citysplit[-1] 
    tempstring = " ".join(citysplit)
    race_by_city.set_value(index,"city",tempstring)   
 ```
# Merging DataFrames
```python
merge_a = pd.merge(race_by_city, poverty_level, on=["city", "state"], how="inner")

merge_b = pd.merge(merge_a, median_house_income, on=["city", "state"], how="inner")

merge_c = pd.merge(merge_b, percentage_complete_highschool, on=[ "city", "state"], how="inner")

merged_data = pd.merge(kill, merge_c, on=["city", "state"], how="inner")
```
```python
# removing any missing values encoded as X or - from merged data
merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]]=merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]].replace("(X)", np.NaN)
merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]]=merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]].replace("-", np.NaN)
merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]]=merged_data[["median_income","poverty_rate", "white", "black", "native_american", "asian" ,"hispanic", "completed_hs"]].astype(float)
#checking dataTypes
merged_data["median_income"] = merged_data["median_income"].astype(float)
merged_data["poverty_rate"] = merged_data["poverty_rate"].astype(float)
merged_data["white"] = merged_data["white"].astype(float)
merged_data["black"] = merged_data["black"].astype(float)
merged_data["native_american"] = merged_data["native_american"].astype(float)
merged_data["asian"] = merged_data["asian"].astype(float)
merged_data["hispanic"] = merged_data["hispanic"].astype(float)
merged_data["completed_hs"] = merged_data["completed_hs"].astype(float)
```
```python
#Creating one-hot encoding
dumy_var = pd.get_dummies(merged_data[["armed","gender","city","state","threat","mental_illness","body_camera"]],drop_first=True)
dumy_var = pd.concat([merged_data,dumy_var], axis=1)
dumy_var.drop(merged_data[["armed","gender","city","state","threat","mental_illness","body_camera","date","manner_of_death","name"]], axis=1, inplace=True)
dumy_var.dropna()
dumy_var.head()
```
# Logistic Regression
# Importing Libraries
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
```
```python
# Define x and y vaeriables
# fitting our model
x = dumy_var.drop("race", axis=1)
y = dumy_var["race"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
logisticmodel = LogisticRegression()
logisticmodel.fit(x_train, y_train)
```
```python
#predict the values based on the test values
predictions = logisticmodel.predict(x_test)
print (classification_report (y_test, predictions))
```
```python
#get the accuracy score
from sklearn.metrics import accuracy_score
logistic_accuracy_score = accuracy_score(y_test, predictions)
logistic_accuracy_score
```
```python
#get the accuracy score
from sklearn.metrics import accuracy_score
logistic_accuracy_score = accuracy_score(y_test, predictions)
logistic_accuracy_score
#Apply GridSearchCV
from sklearn import model_selection
parameters = {"max_iter": [20,30,50], "C": [1.0, 2.0, 3.0]}
gs_logisticmodel = model_selection.GridSearchCV(estimator=logisticmodel, param_grid=parameters, cv=5, scoring="accuracy")
gs_logisticmodel.fit(x_train, y_train)
```
```python
#Get the best parameters 
gs_logisticmodel.best_params_
#Get the accuracy of the model
gs_logisticmodel_accuracy_score = gs_logisticmodel.best_score_
gs_logisticmodel_accuracy_score
```
# Plot the confusion Matrix
```python
import seaborn as sns; sns.set()
confmat = confusion_matrix(y_test, predictions)
sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Pred label')
```
# KNN
```python
#setting our variables 

from sklearn.neighbors import KNeighborsClassifier
x = dumy_var.drop("race", axis=1)
y = dumy_var["race"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(x_train, y_train)

#get the prediction and print the report

predictions = knn_model.predict(x_test)
print(classification_report(y_test, predictions))
#Get the accuracy score of the knn model

knn_accuracy_score = accuracy_score(y_test, predictions)
knn_accuracy_score
```
```python
#calculating the errors for a range of k={1,30}

error = []

for i in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(x_train, y_train)
    i_predictions = knn_model.predict(x_test)
    error.append(np.mean(i_predictions != y_test))
    
#plotting knn errors per k values 
plt.figure(figsize=(15,10))
plt.plot(range(1,30),error,color="blue", marker="x")
plt.title(" Error per K values")
plt.xlabel("k")
plt.ylabel("Error")

#setting k = 12
#printing the predictions report

knn_model = KNeighborsClassifier(n_neighbors=12)
knn_model.fit(x_train, y_train)
predictions = knn_model.predict(x_test)

print(classification_report(y_test, predictions))
```
```python
#Getting the accuracy of our knn model

knn_accuracy_score_iteration = accuracy_score(y_test, predictions)
knn_accuracy_score_iteration
```
```python
#Accuracy score comparaison between the 2 models before and after Grid Search

pre_accuracy = {"Log Regression" : logistic_accuracy_score, "KNN": knn_accuracy_score}


post_accuracy = {"Log Regression" : gs_logisticmodel_accuracy_score, "KNN": knn_accuracy_score_iteration}

x = np.arange(len(pre_accuracy))
ax = plt.subplot(111)
ax.bar(x, pre_accuracy.values(), width=0.2, color='r')
ax.bar(x-0.2,post_accuracy.values(),width=0.2, color='b')
ax.legend(('before','after'))
plt.xticks(x,pre_accuracy.keys())

plt.title("Accuracy scores comparaison")
plt.show()
```
# Random Forest
```python
#True / false to 1/0 
#Using Label ecoder for categorical data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

merged_data["mental_illness"] = merged_data["mental_illness"].astype(int)
merged_data["body_camera"] = kill["body_camera"].astype(int)

le = LabelEncoder()
le.fit(["armed", "race", "gender", "city", "state", "threat"])

#setting our variables 

X = merged_data_log
y = merged_data_log["race"]
X.drop(["race", "date"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Fitting our classifier

rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)
```
```python
#Getting our predictions

rfc_model_pred = rfc_model.predict(X_test)
rfc_model.feature_importances_
#Listing our feature importance

features = pd.DataFrame({"feature_name": merged_data_log.columns, "feature_importance": rfc_model.feature_importances_}) 
features
#Print the classification report

print(classification_report(y_test, rfc_model_pred))
#Get the accuracy score

rfc_model_accuracy_score = accuracy_score(y_test, rfc_model_pred)
rfc_model_accuracy_score
```
```python
#Apply GridSearchCV

params = {"max_depth": [32,44,50],"n_estimators": [15,18,26,32],"min_samples_leaf": [40,50,60],"criterion": ["gini", "entropy"]}

gs_rfc_model = model_selection.GridSearchCV(estimator=rfc_model,param_grid=params,cv=5,scoring="accuracy")

gs_rfc_model.fit(X_train, y_train)
# the best parameters

gs_rfc_model.best_params_
# Get the accuracy score 

gs_rfc_model_accuracy_score = gs_rfc_model.best_score_ 
gs_rfc_model_accuracy_score
```
```python
#Confusion Matrix

mat = confusion_matrix(y_test, rfc_model_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
```
# Accuracy score comparaison between all 3 models before and after Grid Search 
```python
pre_accuracy = {"Random Forest": rfc_model_accuracy_score,"Log Regression" : logistic_accuracy_score, "KNN": knn_accuracy_score}


post_accuracy = {"Random Forest": gs_rfc_model_accuracy_score,"Log Regression" : gs_logisticmodel_accuracy_score, "KNN": knn_accuracy_score_iteration}

x = np.arange(len(pre_accuracy))
ax = plt.subplot(111)
ax.bar(x, pre_accuracy.values(), width=0.2, color='r')
ax.bar(x-0.2,post_accuracy.values(),width=0.2, color='b')
ax.legend(('before','after'))
plt.xticks(x,pre_accuracy.keys())

plt.title("Accuracy scores comparaison")
plt.show()
```
