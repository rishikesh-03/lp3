# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = pd.read_csv("diabetes.csv")
data

# %%
df = pd.DataFrame(data)
df.head()

# %%
df.isnull().sum()

# %%
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

# %% [markdown]
# # Manipulating and Cleaning our dataset

# %%
cols_clean = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']
for i in cols_clean:
    df[i] = df[i].replace(0,np.NaN)
    cols_mean = int(df[i].mean(skipna=True))
    df[i] = df[i].replace(np.NaN, cols_mean)
data1 = df
data1.head().style.highlight_max(color="lightblue").highlight_min(color="red")

# %%
print(data1.describe())

# %%
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

graph = ['Glucose','Insulin','BMI','Age','Outcome']
sns.set()
# print(sns.pairplot(data1[graph],hue='Outcome', diag_kind='kde'))
print(sns.pairplot(data1[graph],hue='Outcome', diag_kind='kde'))

# %%
# for the purpose of simplicity and analysing the most relevent  data , we will select three features of the dataset
# Glucose , Insulin and BMI
# defining variables and features for the dataset for splitting 
# q_cols = ['Glucose','Insulin','BMI','Outcome']
q_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = data1[q_cols]
print(df.head(2))

# %%
# # let's split the data into training and testing datasets
# split = 0.75 # 75% train and 25% test dataset
# total_len = len(df)
# split_df = int(total_len*split)
# train, test = df.iloc[:split_df,0:4],df.iloc[split_df:,0:4] 
# train_x = train[['Glucose','Insulin','BMI']]
# train_y = train['Outcome']
# test_x = test[['Glucose','Insulin','BMI']]
# test_y = test['Outcome']

# Split the data into training and testing datasets
split = 0.75  # 75% train and 25% test dataset
total_len = len(df)
split_df = int(total_len * split)
train, test = df.iloc[:split_df], df.iloc[split_df:]

# Select the columns specified in q_cols for training and testing
train_x = train[q_cols[:-1]]  # Exclude the 'Outcome' column from features
train_y = train['Outcome']    # Target variable
test_x = test[q_cols[:-1]]    # Exclude the 'Outcome' column from features
test_y = test['Outcome']      # Target variable


# %%
a = len(train_x) 
b = len(test_x)
print(' Training data =',a,'\n','Testing data =',b,'\n','Total data length = ',a+b)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def knn(x_train, y_train, x_test, y_test,n):
    n_range = range(1, n)
    results = []
    for n in n_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train, y_train)
        #Predict the response for test dataset
        predict_y = knn.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predict_y)
        #matrix = confusion_matrix(y_test,predict_y)
        #seaborn_matrix = sns.heatmap(matrix, annot = True, cmap="Blues",cbar=True)
        results.append(accuracy)
    return results

# %%
n= 500
output = knn(train_x,train_y,test_x,test_y,n)
n_range = range(1, n)
plt.plot(n_range, output)

# %%
# best k that could optimize this model is between 100 to 200 offering a 77% accuracy
# ideal k value for this dataset should be 150 give or take

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
y_pred = knn(train_x,train_y,test_x,test_y,n)
cnf_matrix = confusion_matrix(test_y, y_pred)

# %%
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
# Define your KNN function to return predictions
def knn2(x_train, y_train, x_test, y_test, n):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)
    # Predict the response for the test dataset
    predict_y = knn.predict(x_test)
    return predict_y

# %%
n = 135
y_pred = knn2(train_x, train_y, test_x, test_y, n)
cnf_matrix = confusion_matrix(test_y, y_pred)

# %%
# Now you can calculate other metrics like accuracy, precision, recall, etc.
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
fbeta = fbeta_score(test_y, y_pred, beta=0.5)

# %%
# Print the confusion matrix and other metrics
print("Confusion Matrix:\n", cnf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("F-beta Score:", fbeta)

# %%
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %%

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path to your dataset
df = pd.read_csv('diabetes.csv')

# Define your feature columns and target column
q_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_col = 'Outcome'

# Split the data into features (X) and target (y)
X = df[q_cols]
y = df[target_col]

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Perform feature scaling (standardization) on the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a K-nearest neighbors (KNN) classifier
k = 5  # You can adjust the value of k
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# %%
p = sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%



