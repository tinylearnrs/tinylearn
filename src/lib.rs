//! Tinylearn is a machine learning library for WebAssembly and `#![no_std]` environments.
//!
//! # Example
//! ```
//! use tinylearn::lm::LinearRegression;
//! use tinylearn::Estimator;
//! use tinylearn::Predictor;
//! use ndarray::array;
//!
//! let xs = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
//! let ys = array![1.0, 2.0, 3.0];
//! let model = LinearRegression { fit_intercept: true };
//! let fitresult = model.fit(&xs, &ys).unwrap();
//! let ps = fitresult.predict(&array![[1.0, 2.0], [3.0, 4.0]]);
//! ```
#![no_std]

use ndarray::Array1;
use ndarray::Array2;
use thiserror::Error;

pub mod lm;

pub trait Predictor {
    fn predict(&self, xs: &Array2<f64>) -> Array1<f64>;
}

#[derive(Error, Debug)]
pub enum EstimatorError {
    #[error("Failed to fit model")]
    FitError,
}

pub trait Estimator {
    type ResultType: Predictor;
    fn fit(&self, xs: &Array2<f64>, ys: &Array1<f64>) -> Result<Self::ResultType, EstimatorError>;
}
