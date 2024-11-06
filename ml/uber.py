# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df=pd.read_csv("uber.csv")

# %%
df.head()

# %%
df.info()

# %%
df.drop(['Unnamed: 0', 'key', 'pickup_datetime'], axis=1, inplace=True)

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
df.isnull().sum()

# %%
df.describe()

# %%
def haversineDistance(lon1, lat1, lon2, lat2):
   R = 6371  # Radius of the Earth in km
   # Converting degrees to radians
   lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
   dlat = lat2 - lat1
   dlon = lon2 - lon1

   a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

   distance = R * c
   return distance


df = df[(df['pickup_longitude'] >= -80) & (df['pickup_longitude'] <= -70)]
df = df[(df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 45)]
df = df[(df['dropoff_longitude'] >= -80) & (df['dropoff_longitude'] <= -70)]
df = df[(df['dropoff_latitude'] >= 40) & (df['dropoff_latitude'] <= 45)]
df['distance_in_KM'] = haversineDistance(df['pickup_longitude'], df['pickup_latitude'],
                                    df['dropoff_longitude'], df['dropoff_latitude'])

# %%
df.head()

# %%
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.show()

# %%
for col in df.columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lb=Q1-1.5*IQR
    ub=Q3+1.5*IQR
    df=df[(df[col]>=lb) & (df[col]<=ub)]

# %%
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.show()

# %%
df.corr()

# %%
sns.heatmap(data=df.corr(), annot=True)
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x=df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance_in_KM']]
y=df['fare_amount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

# %%
model = LinearRegression()
model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_test)

# %%
r2score = r2_score(y_test, y_pred)
print("R2 Score: ", r2score)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)

# %%
from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor()
model1.fit(x_train, y_train)

# %%
y_pred = model1.predict(x_test)

# %%
r2score = r2_score(y_test, y_pred)
print("R2 Score: ", r2score)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)


