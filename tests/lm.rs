use approx::assert_abs_diff_eq;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use wsk::lm;

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
    let model = model.fit(&xs, &ys);
    tracing::info!("model: {:?}", model);
    assert_abs_diff_eq!(
        model.coefficients,
        &array![-0.51499, 0.51175],
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(model.intercept, 19.24392220, epsilon = 1e-8);

    let model = lm::LinearRegression {
        fit_intercept: false,
    };
    let model = model.fit(&xs, &ys);
    tracing::info!("model: {:?}", model);
    assert_abs_diff_eq!(
        model.coefficients,
        &array![1.95896584, -0.20944023],
        epsilon = 1e-7
    );
    assert_abs_diff_eq!(model.intercept, 0.0, epsilon = 1e-8);
}
