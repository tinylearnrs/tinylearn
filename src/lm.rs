use ndarray::Array1;
use ndarray::ArrayD;
use ndarray::Axis;

fn preprocess_data(
    xs: &ArrayD<f64>,
    y: &Array1<f64>,
) -> (ArrayD<f64>, Array1<f64>, Array1<f64>, f64) {
    let n_samples = xs.shape()[0];
    let n_features = xs.shape()[1];
    let mut xs = xs.to_owned();
    let mut y = y.to_owned();

    let mut xs_offset = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        xs_offset[i] = xs.mean_axis(Axis(0)).unwrap()[i];
    }
    for mut row in xs.axis_iter_mut(Axis(0)) {
        for (i, x) in row.iter_mut().enumerate() {
            *x -= xs_offset[i];
        }
    }

    // let xs_offset = xs_offset.mean_axis(Axis(0)).unwrap().to_owned();
    tracing::info!("n_samples: {:?}", n_samples);
    let y_offset = y.sum() / n_samples as f64;
    for y in y.iter_mut() {
        *y -= y_offset;
    }

    (xs, y, xs_offset, y_offset)
}

#[test]
fn test_preprocess_data() {
    tracing_subscriber::fmt::init();

    use ndarray::{Array1, ArrayD, IxDyn};
    let mut reader = csv::Reader::from_path("tests/basic.csv").unwrap();

    // Skip the header row
    let headers = reader.headers().unwrap().clone();

    // Count the number of records to pre-allocate arrays
    let record_count = reader.records().count();
    reader = csv::Reader::from_path("tests/basic.csv").unwrap();
    reader.headers().unwrap(); // Skip headers again

    // Create arrays directly
    let mut y = Array1::<f64>::zeros(record_count);
    let mut xs = ArrayD::<f64>::zeros(IxDyn(&[record_count, headers.len() - 1]));

    for (i, result) in reader.records().enumerate() {
        let record = result.unwrap();
        // Based on the CSV format, y is in column 0, x1 in column 1, x2 in column 2
        y[i] = record[0].parse::<f64>().unwrap();
        for j in 1..record.len() {
            xs[[i, j - 1]] = record[j].parse::<f64>().unwrap();
        }
    }
    for row in xs.axis_iter(Axis(0)) {
        tracing::info!("row: {:?}", row);
    }
    for y in y.iter() {
        tracing::info!("y: {y}");
    }

    let (_xs, _y, xs_offset, y_offset) = preprocess_data(&xs, &y);
    assert_eq!(y_offset, 6.0);
    assert_eq!(xs_offset.into_raw_vec_and_offset().0, &[8.4, 8.6]);
}
