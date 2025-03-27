use approx::assert_abs_diff_eq;
use ndarray::array;
use tinylearn::lm::LogisticRegressionPenalty;
use tinylearn::lm::LogisticRegression;
use tinylearn::Estimator;

#[test]
fn test_logistic_regression() {
    tracing_subscriber::fmt::init();

    // Numbers obtained via scikit-learn:
    // ```py
    // xs = [
    //     [1.0, 2.0],
    //     [5.0, 8.0],
    // ]
    // y = [1, 2]
    // model = LogisticRegression(fit_intercept=True, penalty=None)
    // model.fit(xs, y)
    // ```
    let xs = array![[1.0, 2.0], [5.0, 8.0]];
    let ys = array![1.0, 2.0];
    let model = LogisticRegression {
        fit_intercept: true,
        penalty: LogisticRegressionPenalty::None,
    };
    let fitresult = model.fit(&xs, &ys).unwrap();
    tracing::info!("fitresult: {:?}", fitresult);
    assert_abs_diff_eq!(fitresult.coefficients, &array![4.86734, 0.037258], epsilon = 1e-3);
    assert_abs_diff_eq!(fitresult.intercept, -14.52750, epsilon = 1e-3);
}