# %% [markdown]
# ## Problem Statement: 
# 
# #### Classify the email using the binary classification method. Email Spam detection has two states: a) Normal State Not Spam, b) Abnormal State Spam. Use K-Nearest Neighbors and Support Vector Machine for classification. Analyze their performance.
# Dataset link: The emails.csv dataset on the Kaggle https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('emails.csv')

# %%
df.head()

# %%
df.info()

# %%
df.columns

# %%
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
df = df.drop(['Email No.'], axis=1)

# %%
X = df.drop(['Prediction'], axis=1)
y = df['Prediction']

# print(len(y)) # total
# print(sum(y)) # one count
# print(len(y) - sum(y)) # zero count

from sklearn.preprocessing import scale
X = scale(X)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### KNN Classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cnf = confusion_matrix(y_test, y_pred)
cm = ConfusionMatrixDisplay(confusion_matrix=cnf, display_labels=['Spam', 'Not Spam'])

cm.plot()
plt.show()

# %% [markdown]
# ### SVM 

# %%
from sklearn.svm import SVC

model = SVC(C=4)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
print(classification_report(y_test, y_pred))

# %%
cnf = confusion_matrix(y_test, y_pred)
cm = ConfusionMatrixDisplay(confusion_matrix=cnf, display_labels=['Spam', 'Not Spam'])

cm.plot()
plt.show()


