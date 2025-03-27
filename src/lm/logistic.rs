//! Logistic Regression.

use crate::Estimator;
use crate::Predictor;
use faer::linalg::solvers::SolveLstsqCore;
use faer_ext::*;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use thiserror::Error;

#[derive(Clone, Debug, Hash, PartialEq)]
pub enum LogisticRegressionPenalty {
    L2,
    None,
}

/// Logistic Regression (aka logit, MaxEnt) classifier.
///
/// This class implements regularized logistic regression using the 'liblinear'
/// library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note that
/// regularization is applied by default**. It can handle both dense and sparse
/// input. Use C-ordered arrays or CSR matrices containing 64-bit floats for
/// optimal performance; any other input format will be converted (and copied).
///
/// The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
/// with primal formulation, or no regularization. The 'liblinear' solver
/// supports both L1 and L2 regularization, with a dual formulation only for the
/// L2 penalty. The Elastic-Net regularization is only supported by the 'saga'
/// solver.
///
/// For :term:`multiclass` problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
/// handle multinomial loss. 'liblinear' and 'newton-cholesky' only handle
/// binary classification but can be extended to handle multiclass by using
/// :class:`~sklearn.multiclass.OneVsRestClassifier`.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct LogisticRegression {
    /// Whether to fit the intercept (default: true).
    ///
    /// If set to `false`, no intercept will be used in calculations (i.e. data
    /// is expected to be centered).
    pub fit_intercept: bool,
    /// The penalty (regularization term) to use (default: [LogisticRegressionPenalty::L2]).
    pub penalty: LogisticRegressionPenalty,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            penalty: LogisticRegressionPenalty::L2,
        }
    }
}

/// Result of fitting the Logistic Regression.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegressionResult {
    pub coefficients: Array1<f64>,
    pub intercept: f64,
}

#[derive(Error, Debug)]
pub enum LogisticRegressionError {
    #[error("Failed to fit model")]
    FitError,
}

fn unique_values(data: &mut [f64]) -> &[f64] {
    if data.is_empty() {
        return &[];
    }
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut write = 1;
    for read in 1..data.len() {
        if data[read] != data[write - 1] {
            data[write] = data[read];
            write += 1;
        }
    }
    &data[..write]
}

fn logistic_regression_path(xs: &Array2<f64>, ys: &Array1<f64>, class: f64) -> f64 {
    todo!()
}

impl Estimator for LogisticRegression {
    type T = LogisticRegressionResult;
    type E = LogisticRegressionError;
    #[allow(unused)]
    fn fit(&self, xs: &Array2<f64>, ys: &Array1<f64>) -> Result<Self::T, Self::E> {
        let penalty;
        let c_;
        match self.penalty {
            LogisticRegressionPenalty::L2 => {}
            LogisticRegressionPenalty::None => {
                c_ = core::f64::MAX;
                penalty = LogisticRegressionPenalty::L2;
            }
        };
        let mut ys_ = ys.clone().as_slice_mut().unwrap().to_vec();
        let mut classes = unique_values(&mut ys_).to_vec();
        // let max_squared_sum = None;
        let mut n_classes = classes.len();

        if n_classes == 2 {
            n_classes = 1;
            classes = classes[1..].to_vec();
        }

        let mut coefs = Array2::<f64>::zeros((n_classes, xs.ncols()));
        for (i, c) in classes.iter().enumerate() {
            coefs[i] = logistic_regression_path(&xs, &ys, *c);
        }

        let intercepts;
        if self.fit_intercept {
            intercepts = coefs.first().unwrap();
            coefs = coefs.slice(s![1..]).to_owned();
        } else {
            intercepts = Array1::zeros(n_classes);
        }

        todo!()
    }
}

impl Predictor for LogisticRegressionResult {
    fn predict(&self, xs: &Array2<f64>) -> Array1<f64> {
        xs.dot(&self.coefficients) + self.intercept
    }
}
