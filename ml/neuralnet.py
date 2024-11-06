# %% [markdown]
# ## Problem Statement:
# 
# #### Given a bank customer, build a neural network-based classifier that can determine whetherthey will leave or not in the next 6 months.
# Dataset Description: The case study is from an open-source dataset from Kaggle.The dataset contains 10,000 sample points with 14 distinct features such as CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, etc.
# Link to the Kaggle project: https://www.kaggle.com/barelydedicated/bank-customer-churn- modeling Perform following steps:
# 1. Read the dataset.
# 2. Distinguish the feature and target set and divide the data set into training and test sets.
# 3. Normalize the train and test data.
# 4. Initialize and build the model. Identify the points of improvement and implement the same.
# Print the accuracy score and confusion matrix 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('Churn_Modelling.csv')

# %%
df.head()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.dtypes

# %%
df.columns

# %%
# Feature Selection
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1)

# %%
def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()

# %%
df_churn_exited = df[df['Exited']==1]['Tenure']
df_churn_not_exited = df[df['Exited']==0]['Tenure']

# %%
visualization(df_churn_exited, df_churn_not_exited, "Tenure")

# %%
df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']

# %%
visualization(df_churn_exited2, df_churn_not_exited2, "Age")

# %%
states = pd.get_dummies(df['Geography'])
gender = pd.get_dummies(df['Gender'])

# %%
df = pd.concat([df,gender,states], axis = 1)

# %%
df.head()

# %%
X = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Female','Male','France','Germany','Spain']]
y = df['Exited']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# %%
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense

# %%
classifier = Sequential()

classifier.add(Dense(activation='relu', input_dim=13, units=6, kernel_initializer='uniform'))
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

# %%
classifier.fit(X_train,y_train,batch_size=10,epochs=50)

# %%
y_pred =classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# %%
from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test, y_pred))

# %%
from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cnf}")

# %%
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc}\n")


