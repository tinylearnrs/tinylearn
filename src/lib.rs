//! Tinylearn is a machine learning library for WebAssembly and `#![no_std]` environments.
//!
//! # Example
//! ```
//! use tinylearn::lm::LinearRegression;
//! use ndarray::array;
//!
//! let xs = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
//! let ys = array![1.0, 2.0, 3.0];
//! let model = LinearRegression { fit_intercept: true };
//! let fit_result = model.fit(&xs, &ys);
//! ```
#![no_std]

pub mod lm;
