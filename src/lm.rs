use approx::assert_abs_diff_eq;
use ndarray::Array1;
use ndarray::ArrayD;
use ndarray::Axis;

fn preprocess_data(
    xs: &ArrayD<f64>,
    ys: &Array1<f64>,
) -> (ArrayD<f64>, Array1<f64>, ArrayD<f64>, f64) {
    let mut xs = xs.to_owned();
    let mut ys = ys.to_owned();

    let xs_offset = xs.mean_axis(Axis(0)).unwrap();
    for mut row in xs.axis_iter_mut(Axis(0)) {
        for (i, x) in row.iter_mut().enumerate() {
            *x -= xs_offset[i];
        }
    }

    let y_offset = ys.mean().unwrap();
    for y in ys.iter_mut() {
        *y -= y_offset;
    }

    (xs, ys, xs_offset, y_offset)
}

#[test]
fn test_preprocess_data() {
    tracing_subscriber::fmt::init();

    use ndarray::{Array1, ArrayD, IxDyn};
    let mut reader = csv::Reader::from_path("tests/basic.csv").unwrap();

    let headers = reader.headers().unwrap().clone();

    // Count the number of records to pre-allocate arrays
    let record_count = reader.records().count();
    reader = csv::Reader::from_path("tests/basic.csv").unwrap();

    let mut ys = Array1::<f64>::zeros(record_count);
    let mut xs = ArrayD::<f64>::zeros(IxDyn(&[record_count, headers.len() - 1]));

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
    for y in ys.iter() {
        tracing::info!("y: {y}");
    }

    let (xs, ys, xs_offset, y_offset) = preprocess_data(&xs, &ys);
    assert_eq!(y_offset, 6.);
    assert_eq!(xs_offset.into_raw_vec_and_offset().0, &[8.4, 8.6]);
    assert_eq!(ys.into_raw_vec_and_offset().0, &[-5., -2., 1., 2., 4.]);
    let mut expected_xs = ArrayD::zeros(IxDyn(&[5, 2]));
    expected_xs[[0, 0]] = -5.4;
    expected_xs[[0, 1]] = -4.6;
    expected_xs[[1, 0]] = -2.4;
    expected_xs[[1, 1]] = -3.6;
    expected_xs[[2, 0]] = 0.6;
    expected_xs[[2, 1]] = 0.4;
    expected_xs[[3, 0]] = 3.6;
    expected_xs[[3, 1]] = 2.4;
    expected_xs[[4, 0]] = 3.6;
    expected_xs[[4, 1]] = 5.4;
    assert_abs_diff_eq!(xs, expected_xs, epsilon = 1e-6);
}
