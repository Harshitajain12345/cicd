import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def sample_data():
    df = pd.read_csv('tips.csv')
    lb = LabelEncoder()
    for col in ['sex', 'smoker', 'day', 'time']:
        df[col] = lb.fit_transform(df[col])
    return df

def test_scaling(sample_data):
    """
    Test if StandardScaler correctly scales the features.
    """
    df = sample_data.copy()
    x = df.drop(columns=['total_bill'])
    sc = StandardScaler()
    x_sc = sc.fit_transform(x)
    x_sc_df = pd.DataFrame(x_sc, columns=x.columns)

    non_zero_var_columns = x.columns[x_sc_df.std() > 0]
    assert np.isclose(x_sc_df[non_zero_var_columns].mean().mean(), 0, atol=1e-1), "Scaling mean is not approximately 0"
    assert np.isclose(x_sc_df[non_zero_var_columns].std().mean(), 1, atol=1e-1), "Scaling standard deviation is not approximately 1"

def test_linear_regression(sample_data):
    """
    Test if LinearRegression correctly fits and predicts.
    """
    df = sample_data.copy()
    x = df.drop(columns=['total_bill'])
    y = df['total_bill']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    assert not np.isnan(r2), "R² score is nan, possibly due to insufficient data"
    assert r2 >= 0, "R² score should be non-negative for a valid regression model"
