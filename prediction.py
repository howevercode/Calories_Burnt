import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

calories= pd.read_csv("calories.csv")

exercise= pd.read_csv("exercise.csv")

data= pd.merge('calories', 'exercise', on= 'User_ID')

data.describe()

sns.set()
sns.countplot(data['Gender'])

sns.displot(data['Age'])

caorrelation= data.corr()

X= data.drop(columns=['UserId', 'Calories'])
y= data['Calories']

X_train, X_test, y_train, y_test= train_test_split= (X, y, test_size= 0.2, random_state= 42)

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(), ['Gender']),
    ('num', StandardScaler(), ['Age', 'Height', 'Weight', 'Duration','Heart_Rate', 'Body_Temp'])
], remainder='passthrough')
"""OrdinalEncoder converts categories (like 'Male', 'Female') into numbers (like 0, 1)
   StandardScaler standardizes each feature by removing the mean and scaling to unit variance (z-score normalization).
   remainder='passthrough tells the transformer to keep all other columns unchanged ""

pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', XGBRegressor())]
)


""
preprocessor is your ColumnTransformer that will transform the raw input data so it’s model-ready.

('model', XGBRegressor()) the actual machine learning model: XGBoost regressor, 
which is great for regression problems and handles both linear and non-linear patterns.

""

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(pipeline, X, y, cv = kfold, scoring = 'r2')

cv_results.mean()

with open('pipeline_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)