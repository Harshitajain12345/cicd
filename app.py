import numpy as np
import pandas as pd
df = pd.read_csv('tips.csv')
df.head(2)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])
df.head()
x = df.drop(columns = ['total_bill'])
y = df['total_bill']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train)
y_pred = lr.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test , y_pred)