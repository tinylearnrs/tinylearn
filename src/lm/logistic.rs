//! Logistic Regression.

use crate::Estimator;
use crate::Predictor;
use faer::linalg::solvers::SolveLstsqCore;
use faer_ext::*;

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