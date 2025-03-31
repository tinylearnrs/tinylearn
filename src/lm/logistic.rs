//! Logistic Regression.

use core::f64;

use crate::Estimator;
use crate::Predictor;
use argmin::core::CostFunction;
use argmin::core::Error as ArgminError;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::LineSearch;
use argmin::core::State;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::condition::WolfeCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use thiserror::Error;

#[derive(Clone, Debug, Hash, PartialEq)]
pub enum LogisticRegressionPenalty {
    L2,
    None,
}

#[derive(Clone, Debug)]
struct LogisticRegressionProblem {
    xs: Array2<f64>,
    ys: Array1<f64>,
    fit_intercept: bool,
    penalty: LogisticRegressionPenalty,
    lambda: f64,
}

impl LogisticRegressionProblem {
    fn get_weights_intercept(
        &self,
        params: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64), ArgminError> {
        let w = params.slice(s![..-1]).to_owned();
        let b = params[params.len() - 1];
        Ok((w, b))
    }
}

impl CostFunction for LogisticRegressionProblem {
    type Param = Array1<f64>;
    type Output = f64;

    /// Calculates the regularized logistic loss (negative log-likelihood).
    /// Cost = (1/m) * sum[ log(1 + exp(-z_i)) if y_i=0 else log(1 + exp(z_i)) ] + Regularization
    /// A more stable form (log-sum-exp):
    /// Cost = (1/m) * sum[ max(z_i, 0) - z_i*y_i + log(1 + exp(-abs(z_i))) ] + Regularization
    /// where z_i = w^T * x_i + b
    /// Regularization = (lambda / 2) * ||w||^2   (for L2)
    fn cost(&self, params: &Self::Param) -> Result<Self::Output, ArgminError> {
        let m = self.xs.nrows();
        if m == 0 {
            return Ok(0.0); // No data, no cost
        }
        let m_f64 = m as f64;

        // 1. Separate weights (w) and intercept (b)
        // Use map_err to convert our custom error to ArgminError
        let (w, b) = self
            .get_weights_intercept(params)
            .map_err(ArgminError::from)?;

        // 2. Calculate linear prediction: z = X * w + b
        // z shape: (m,)
        let mut z = self.xs.dot(&w);
        if self.fit_intercept {
            z.mapv_inplace(|val| val + b); // Add intercept b to each element
        }

        // 3. Calculate logistic loss using numerically stable log-sum-exp trick
        // term1 = max(z_i, 0)
        // term2 = z_i * y_i
        // term3 = log(1 + exp(-abs(z_i)))
        // loss_samples = term1 - term2 + term3
        let term1 = z.mapv(|zi| zi.max(0.0));
        let term2 = z
            .iter()
            .zip(self.ys.iter())
            .map(|(zi, yi)| zi * yi)
            .collect::<Array1<f64>>();
        // Calculate term3 carefully: exp(-abs(z)) -> mapv(ln(1+x))
        let term3 = z.mapv(|zi| (-zi.abs()).exp()).mapv(|v| (1.0 + v).ln());

        // Sum over all samples and average
        let data_loss = (term1 - term2 + term3).sum() / m_f64;

        // 4. Add regularization term (only for weights w, not intercept b)
        let reg_cost = match self.penalty {
            LogisticRegressionPenalty::L2 => {
                // Regularization = (lambda / 2) * ||w||^2
                // Note: ||w||^2 = w.dot(&w)
                0.5 * self.lambda * w.dot(&w)
            }
            LogisticRegressionPenalty::None => 0.0,
        };

        // Total cost
        Ok(data_loss + reg_cost)
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl Gradient for LogisticRegressionProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    /// Calculate the gradient of the regularized logistic loss.
    ///
    /// grad_w = (1/m) * X^T * (sigmoid(z) - y) + lambda * w
    /// grad_b = (1/m) * sum(sigmoid(z) - y)
    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let m = self.xs.nrows();
        if m == 0 {
            // Return zero gradient of the correct size
            return Ok(Array1::zeros(params.len()));
        }
        let m_f64 = m as f64;
        let n_features = self.xs.ncols();

        // 1. Separate weights (w) and intercept (b)
        let (w, b) = self
            .get_weights_intercept(params)
            .map_err(ArgminError::from)?;

        // 2. Calculate linear prediction: z = X * w + b
        let mut z = self.xs.dot(&w);
        if self.fit_intercept {
            z.mapv_inplace(|val| val + b); // Add intercept b
        }

        // 3. Calculate predictions (probabilities): h = sigmoid(z)
        let h = z.mapv(sigmoid);

        // 4. Calculate error: error = h - y
        let error = h
            .iter()
            .zip(self.ys.iter())
            .map(|(hi, yi)| hi - yi)
            .collect::<Array1<f64>>();
        // error shape: (m,)

        // 5. Calculate gradient w.r.t. weights (grad_w)
        // grad_w = (1/m) * X^T * error + [regularization term]
        // X^T is (n_features, m), error is (m,) -> result is (n_features,)
        let mut grad_w = self.xs.t().dot(&error) / m_f64;

        // Add L2 regularization term to grad_w (if applicable)
        // Regularization term: lambda * w
        if self.penalty == LogisticRegressionPenalty::L2 {
            // Use scaled_add: grad_w = 1.0 * grad_w + self.lambda * w
            grad_w.scaled_add(self.lambda, &w);
        }

        // 6. Calculate gradient w.r.t. intercept (grad_b) (if applicable)
        // grad_b = (1/m) * sum(error)
        let grad_b = if self.fit_intercept {
            error.sum() / m_f64
        } else {
            // If no intercept, this part of the gradient doesn't exist
            // but we need to handle it when combining below.
            0.0 // Placeholder, won't be used if fit_intercept is false
        };

        // 7. Combine gradients into a single vector matching `params` shape
        let final_grad = if self.fit_intercept {
            // Stack grad_w and grad_b
            let mut grad = Array1::zeros(n_features + 1);
            grad.slice_mut(s![..n_features]).assign(&grad_w);
            grad[n_features] = grad_b;
            grad
            // Alternative using stack (might be slightly less efficient due to view creation)
            // ndarray::stack(Axis(0), &[grad_w.view(), ArrayView1::from_shape((1,), &[grad_b]).unwrap()])
            //     .unwrap() // stack can fail if shapes mismatch
            //     .into_shape((n_features + 1,)).unwrap() // Reshape back to 1D
        } else {
            // Gradient is just grad_w
            grad_w
        };

        Ok(final_grad)
    }
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
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegression {
    /// Whether to fit the intercept (default: true).
    ///
    /// If set to `false`, no intercept will be used in calculations (i.e. data
    /// is expected to be centered).
    pub fit_intercept: bool,
    /// The penalty (regularization term) to use (default: [LogisticRegressionPenalty::L2]).
    pub penalty: LogisticRegressionPenalty,
    /// Inverse of the regularization strength (default: 1.0).
    ///
    /// Must be a positive number. Like in support vector machines, smaller
    /// values specify stronger regularization.
    pub c: f64,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            penalty: LogisticRegressionPenalty::L2,
            c: 1.0,
        }
    }
}

/// Result of fitting the Logistic Regression.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticRegressionResult {
    pub coefficients: Array2<f64>,
    pub intercepts: Array1<f64>,
}

#[derive(Error, Debug)]
pub enum LogisticRegressionError {
    #[error("Failed to fit model")]
    FitError,
}

fn unique(data: &mut [f64]) -> &[f64] {
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

#[test]
fn test_unique_values() {
    // ```py
    // ys = [1, 2]
    // print(np.unique(ys))
    // ```
    let mut data = [2.0, 1.0, 2.0];
    let unique = unique(&mut data);
    assert_eq!(unique, &[1.0, 2.0]);
}

#[allow(unused)]
struct LogisticRegressionPathArgs<'a> {
    xs: &'a Array2<f64>,
    ys: &'a Array1<f64>,
    class: f64,
    classes: &'a [f64],
    fit_intercept: bool,
    c: f64,
}

fn minimize(
    xs: &Array2<f64>,
    ys: &Array1<f64>,
    l2_reg_strength: f64,
    fit_intercept: bool,
) -> Array1<f64> {
    let problem = LogisticRegressionProblem {
        xs: xs.to_owned(),
        ys: ys.to_owned(),
        fit_intercept,
        penalty: LogisticRegressionPenalty::L2,
        lambda: l2_reg_strength,
    };
    let param_len = if fit_intercept {
        xs.ncols() + 1
    } else {
        xs.ncols()
    };
    let init_param = Array1::<f64>::zeros(param_len);

    let ls: MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64> =
        MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

    // Tolerance for loss function.
    let ftol = 64.0 * core::f64::EPSILON;
    let gtol = 0.0001;

    // argmin in the tests uses 10 when comparing to scipy.
    // argmin/crates/argmin/src/tests.rs
    let m = 10;
    let solver = LBFGS::new(ls, m)
        .with_tolerance_cost(ftol)
        .unwrap()
        .with_tolerance_grad(gtol)
        .unwrap();

    let max_iter = 100;
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(init_param).max_iters(max_iter))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    result.state.get_best_param().unwrap().to_owned()
}

#[test]
fn test_minimize() {
    // ```py
    // xs = [
    //     [1.0, 2.0],
    //     [5.0, 8.0],
    // ]
    // ys = [1, 2]
    let mut xs = Array2::<f64>::zeros((2, 2));
    xs[[0, 0]] = 1.0;
    xs[[0, 1]] = 2.0;
    xs[[1, 0]] = 5.0;
    xs[[1, 1]] = 8.0;
    let mut ys = Array1::<f64>::zeros(2);
    ys[0] = 0.0;
    ys[1] = 1.0;
    let l2_reg_strength = 0.0;
    let fit_intercept = true;
    let w_min = minimize(&xs, &ys, l2_reg_strength, fit_intercept);
    let mut expected = Array1::<f64>::zeros(3);
    expected[0] = 4.8673412;
    expected[1] = 0.03725813;
    expected[2] = -14.52750733;
    assert_eq!(w_min, expected);
}

fn logistic_regression_path(args: &LogisticRegressionPathArgs) -> Array1<f64> {
    let n_samples = args.xs.nrows();
    let mut n_features = args.xs.ncols();
    let classes = args.classes;
    let pos_class = classes.first().unwrap();

    // v1.6.1 _logistic.py#375
    // For binary problems coef.shape[0] should be 1
    let n_classes = if classes.len() <= 2 { 1 } else { todo!() };
    let coef = Array1::<f64>::zeros(n_classes);
    if coef.len() != n_classes {
        panic!("coef.shape[0] should be 1");
    }
    // v1.6.1 _logistic.py#316
    let y_bin = args.ys.map(|y| if *y == *pos_class { 1.0 } else { 0.0 });
    // v1.6.1 _logistic.py#363
    // w0 is used for warm_start.
    // let mut w0 = Array2::<f64>::ones((n_classes, n_features));
    // v1.6.1 _logistic.py#398
    // if n_classes == 1 {
    //     w0.slice_mut(s![0, coef.len()]).assign(&-coef.clone());
    //     w0.slice_mut(s![1, coef.len()]).assign(&coef);
    // }
    let sw_sum = n_samples;
    // v1.6.1 _logistic.py#423
    let target = y_bin;

    // v1.6.1 _logistic.py#348
    if args.fit_intercept {
        n_features += 1;
    }
    let xs = if args.fit_intercept {
        let shape = (n_samples, n_features);
        let mut x_augmented = Array2::<f64>::zeros(shape);
        x_augmented
            .slice_mut(s![.., -1])
            .assign(&Array1::<f64>::ones(n_samples));
        x_augmented.slice_mut(s![.., ..-1]).assign(&args.xs);
        x_augmented
    } else {
        todo!()
    };

    let l2_reg_strength = 1.0 / (args.c * sw_sum as f64);
    let w_min = minimize(&xs, &target, l2_reg_strength, args.fit_intercept);
    w_min
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
        let mut classes = unique(&mut ys_).to_vec();
        // let max_squared_sum = None;
        let mut n_classes = classes.len();

        if n_classes == 2 {
            classes = classes[1..].to_vec();
        }

        let n_features = if self.fit_intercept {
            xs.ncols() + 1
        } else {
            xs.ncols()
        };
        let mut out = Array2::<f64>::zeros((n_classes, n_features));
        for (i, c) in classes.iter().enumerate() {
            let args = LogisticRegressionPathArgs {
                xs: &xs,
                ys: &ys,
                class: *c,
                classes: &classes,
                fit_intercept: self.fit_intercept,
                c: self.c,
            };
            let val = logistic_regression_path(&args);
            out.slice_mut(s![i, ..]).assign(&val);
        }

        let intercepts = if self.fit_intercept {
            // self.coef[:, -1]
            out.slice(s![.., -1]).to_owned()
        } else {
            Array1::zeros(n_classes)
        };
        let coefficients = if self.fit_intercept {
            // self.coef[:, :-1]
            out.slice(s![.., ..-1]).to_owned()
        } else {
            out
        };

        Ok(LogisticRegressionResult {
            coefficients,
            intercepts,
        })
    }
}

impl Predictor for LogisticRegressionResult {
    fn predict(&self, _xs: &Array2<f64>) -> Array1<f64> {
        todo!()
    }
}
