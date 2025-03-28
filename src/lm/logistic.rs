//! Logistic Regression.

use crate::Estimator;
use crate::Predictor;
use faer::linalg::solvers::SolveLstsqCore;
use faer_ext::*;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Slice;
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
    pub intercepts: Array1<f64>,
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

#[allow(unused)]
struct LogisticRegressionPathArgs<'a> {
    xs: &'a Array2<f64>,
    ys: &'a Array1<f64>,
    class: f64,
    classes: &'a [f64],
}

fn loss_gradient(
    w: &Array1<f64>,
    xs: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
) -> (f64, Array1<f64>) {
    let n_samples = xs.nrows();
    let n_features = xs.ncols();

    // Initialize gradient vector and loss
    let mut grad = Array1::<f64>::zeros(n_features);
    let mut loss = 0.0;

    // Calculate predictions and loss for binary classification
    for i in 0..n_samples {
        // Calculate linear prediction: z = X * w
        let mut z = 0.0;
        for j in 0..n_features {
            z += xs[[i, j]] * w[j];
        }

        // Apply sigmoid function: p = 1 / (1 + exp(-z))
        let p = 1.0 / (1.0 + (-z).exp());

        // Binary cross-entropy loss
        // L = -y*log(p) - (1-y)*log(1-p)
        loss -= y[i] * (p + 1e-10).ln() + (1.0 - y[i]) * (1.0 - p + 1e-10).ln();

        // Gradient: X^T * (p - y)
        let diff = p - y[i];
        for j in 0..n_features {
            grad[j] += xs[[i, j]] * diff;
        }
    }

    // Add L2 regularization term to loss and gradient if alpha > 0
    if alpha > 0.0 {
        for j in 0..n_features {
            loss += 0.5 * alpha * w[j] * w[j];
            grad[j] += alpha * w[j];
        }
    }

    // Average loss and gradient
    loss /= n_samples as f64;
    for j in 0..n_features {
        grad[j] /= n_samples as f64;
    }

    (loss, grad)
}

#[allow(unused)]
fn minimize(xs: &Array2<f64>, c: f64) -> f64 {
    let sw_sum = 99.0;
    let l2_reg_strength = 1.0 / (c * sw_sum);
    let loss = 0.0;
    // v1.6.1 _logistic.py#429.
    let f = loss_gradient;
    // v1.6.1 _logistic.py#471.

    // Use BFGS optimization to minimize the logistic regression loss function
    let n_features = xs.ncols();
    let mut w0 = Array1::<f64>::zeros(n_features);

    // Define the objective function that returns the loss
    let f = |w: &Array1<f64>| {
        let (loss, _) = loss_gradient(w, xs, &Array1::<f64>::ones(xs.nrows()), l2_reg_strength);
        loss
    };

    // Define the gradient function
    let g = |w: &Array1<f64>| {
        let (_, grad) = loss_gradient(w, xs, &Array1::<f64>::ones(xs.nrows()), l2_reg_strength);
        grad
    };

    // Run BFGS optimization
    match crate::bfgs::bfgs(w0, f, g) {
        Ok(w_min) => {
            tracing::debug!("BFGS optimization converged to w_min: {:?}", w_min);
            let loss = f(&w_min);
            loss
        }
        Err(e) => {
            tracing::warn!("BFGS optimization failed: {:?}", e);
            f64::INFINITY // Return infinity to indicate failure
        }
    }
}

#[allow(unused)]
fn logistic_regression_path(args: &LogisticRegressionPathArgs) -> f64 {
    let n_samples = args.xs.nrows();
    let n_features = args.xs.ncols();
    let classes = args.classes;
    let pos_class = classes.first().unwrap();

    // For binary problems coef.shape[0] should be 1
    let n_classes = if classes.len() == 2 { 1 } else { todo!() };
    let mut coef = Array1::<f64>::zeros(n_classes);
    if coef.len() != n_classes {
        panic!("coef.shape[0] should be 1");
    }
    // v1.6.1 _logistic.py#316
    let y_bin = args.ys.map(|y| if *y == *pos_class { 1.0 } else { 0.0 });
    // v1.6.1 _logistic.py#423
    let target = y_bin;

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
            LogisticRegressionPenalty::L2 => {
                c_ = 1.0;
            }
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

        let mut coefs = Array1::<f64>::zeros(n_classes);
        for (i, c) in classes.iter().enumerate() {
            let args = LogisticRegressionPathArgs {
                xs: &xs,
                ys: &ys,
                class: *c,
                classes: &classes,
            };
            let val = logistic_regression_path(&args);
            coefs[i] = val;
        }

        let intercepts;
        if self.fit_intercept {
            // self.coef[:, -1]
            intercepts = coefs.slice(s![..-1]).to_owned();
            // self.coef[:, :-1]
            coefs = coefs.slice(s![..-1]).to_owned();
        } else {
            intercepts = Array1::zeros(n_classes);
        }

        Ok(LogisticRegressionResult {
            coefficients: coefs,
            intercepts,
        })
    }
}

impl Predictor for LogisticRegressionResult {
    fn predict(&self, xs: &Array2<f64>) -> Array1<f64> {
        xs.dot(&self.coefficients) + &self.intercepts
    }
}
