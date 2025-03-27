use approx::assert_abs_diff_eq;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use tinylearn::lm;
use tinylearn::Estimator;
use tinylearn::Predictor;

#[test]
fn test_linear_regression() {
    tracing_subscriber::fmt::init();

    let mut reader = csv::Reader::from_path("tests/basic.csv").unwrap();

    let headers = reader.headers().unwrap().clone();

    let record_count = reader.records().count();
    reader = csv::Reader::from_path("tests/basic.csv").unwrap();

    let mut ys = Array1::<f64>::zeros(record_count);
    let mut xs = Array2::<f64>::zeros((record_count, headers.len() - 1));

    for (i, result) in reader.records().enumerate() {
        let record = result.unwrap();
        ys[i] = record[0].parse::<f64>().unwrap();
        for j in 1..record.len() {
            xs[[i, j - 1]] = record[j].parse::<f64>().unwrap();
        }
    }
    for row in xs.axis_iter(Axis(0)) {
        tracing::info!("row: {:?}", row);
    }
    tracing::info!("ys: {:?}", ys);

    let model = lm::LinearRegression {
        fit_intercept: true,
    };
    let fitresult = model.fit(&xs, &ys).unwrap();
    tracing::info!("fitresult: {:?}", fitresult);
    assert_abs_diff_eq!(
        fitresult.coefficients,
        &array![-0.51499, 0.51175],
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(fitresult.intercept, 19.24392220, epsilon = 1e-8);

    let model2 = lm::LinearRegression {
        fit_intercept: false,
    };
    assert_ne!(model, model2);
    let fitresult2 = model2.fit(&xs, &ys).unwrap();
    tracing::info!("fitresult2: {:?}", fitresult2);
    assert_abs_diff_eq!(
        fitresult2.coefficients,
        &array![1.95896584, -0.20944023],
        epsilon = 1e-7
    );
    assert_abs_diff_eq!(fitresult2.intercept, 0.0, epsilon = 1e-8);
    let ps = fitresult2.predict(&array![[1.0, 2.0], [3.0, 4.0]]);
    assert_abs_diff_eq!(ps, &array![1.54008, 5.03913], epsilon = 1e-4);
}
