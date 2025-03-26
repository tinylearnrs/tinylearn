#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "scikit-learn==1.6.1",
#     "numpy==2.2.0",
# ]
# ///

from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import _preprocess_data
import os
from numpy import genfromtxt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'basic.csv')

data = genfromtxt(csv_path, delimiter=',')
xs = data[1:, 1:3]
y = data[1:, 0]

print("y: ", y)
print("xs: ", xs)
xs, y, xs_offset, y_offset, xs_scale = _preprocess_data(xs, y, fit_intercept=True)

print("xs_offset:", xs_offset)
print("y_offset:", y_offset)
print("xs_scale:", xs_scale)
print("preprocessed xs: ", xs)
print("preprocessed y: ", y)

model = LinearRegression(fit_intercept=True)
model.fit(xs, y)

print("coefficients:", model.coef_)
print("intercept:", round(model.intercept_, 2))

