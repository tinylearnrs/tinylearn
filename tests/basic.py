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
import csv
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'basic.csv'), 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    # Skip the header row
    header = data[0]
    data = data[1:]

xs = np.array([[float(x) for x in row[:-1]] for row in data])
y = np.array([float(row[-1]) for row in data])

xs, y, xs_offset, y_offset, xs_scale = _preprocess_data(xs, y, fit_intercept=True)

print("xs_offset:", xs_offset)
print("y_offset:", y_offset)
print("xs_scale:", xs_scale)

model = LinearRegression(fit_intercept=True)
model.fit(xs, y)

print("coefficients:", model.coef_)
print("intercept:", round(model.intercept_, 2))

