use ndarray::Array2;
use ndarray::Array1;
use ndarray::ArrayD;

fn preprocess_data(xs: &ArrayD<f64>, y: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut xs_offset = Array1::<f64>::zeros(xs.shape()[1]);
    let mut y_offset = 0.0;
    let mut xs_scale = Array1::<f64>::zeros(xs.shape()[1]);

    for i in 0..xs.shape()[0] {
        for j in 0..xs.shape()[1] {
            xs_offset[j] += xs[[i, j]];
        }
        y_offset += y[i];
    }
    (xs, y, xs_offset, y_offset)
}

#[test]
fn test_preprocess_data() {
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
            xs[[i, j-1]] = record[j].parse::<f64>().unwrap();
        }
    }

    let (xs, y, xs_offset, y_offset) = preprocess_data(&xs, &y);
    let mut expected = Array1::<f64>::zeros(2);
    expected[0] = 6.0;
    expected[1] = 8.4;
    assert_eq!(xs_offset, expected);
}