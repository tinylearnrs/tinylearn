use approx::assert_abs_diff_eq;
use ndarray::array;
use tinylearn::lm::LogisticRegression;
use tinylearn::lm::LogisticRegressionPenalty;
use tinylearn::Estimator;

#[test]
fn test_logistic_regression() {
    tracing_subscriber::fmt::init();

    // Numbers obtained via scikit-learn:
    // ```py
    // import numpy as np
    // from sklearn.linear_model import LogisticRegression
    //
    // xs = np.array([
    //     [1.0, 2.0],
    //     [5.0, 8.0],
    // ])
    // ys = [1, 2]
    // model = LogisticRegression(fit_intercept=True, penalty=None)
    // model.fit(xs, ys)
    // print(model.coef_)
    // print(model.intercept_)
    // ```
    // See also test minimize in `logistic.rs`.
    let xs = array![[1.0, 2.0], [5.0, 8.0]];
    let ys = array![1.0, 2.0];
    let model = LogisticRegression {
        fit_intercept: true,
        penalty: LogisticRegressionPenalty::None,
        ..Default::default()
    };
    let fitresult = model.fit(&xs, &ys).unwrap();
    tracing::info!("fitresult: {:?}", fitresult);
    assert_abs_diff_eq!(
        fitresult.coefficients,
        &array![[4.86734, 0.037258]],
        epsilon = 0.5
    );
    assert_abs_diff_eq!(fitresult.intercepts, &array![-14.52750], epsilon = 1.5);
}
