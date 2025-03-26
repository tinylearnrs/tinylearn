use faer_ext::*;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
struct PreprocessedData {
    xs: Array2<f64>,
    ys: Array1<f64>,
    xs_offset: Array1<f64>,
    y_offset: f64,
}

fn preprocess_data(xs: &Array2<f64>, ys: &Array1<f64>, fit_intercept: bool) -> PreprocessedData {
    let mut xs = xs.to_owned();
    let mut ys = ys.to_owned();

    if fit_intercept {
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

        PreprocessedData {
            xs,
            ys,
            xs_offset,
            y_offset,
        }
    } else {
        todo!("fit_intercept is false")
    }
}

#[test]
fn test_preprocess_data() {
    fn round(x: f64, precision: u32) -> f64 {
        let factor = 10.0_f64.powi(precision as i32);
        (x * factor).round() / factor
    }
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

    let preprocessed = preprocess_data(&xs, &ys, true);
    assert_eq!(preprocessed.y_offset, 18.6);
    assert_eq!(
        preprocessed.xs_offset.into_raw_vec_and_offset().0,
        &[9.2, 8.]
    );
    let ys = preprocessed.ys.map(|y| round(*y, 3));
    assert_eq!(
        ys.into_raw_vec_and_offset().0,
        &[-0.6, 1.4, -1.6, -0.6, 1.4]
    );
    let mut expected_xs = Array2::zeros((5, 2));
    expected_xs[[0, 0]] = -4.2;
    expected_xs[[0, 1]] = -4.;
    expected_xs[[1, 0]] = -3.2;
    expected_xs[[1, 1]] = -3.;
    expected_xs[[2, 0]] = -0.2;
    expected_xs[[2, 1]] = -2.;
    expected_xs[[3, 0]] = 4.8;
    expected_xs[[3, 1]] = 3.;
    expected_xs[[4, 0]] = 2.8;
    expected_xs[[4, 1]] = 6.;
    approx::assert_abs_diff_eq!(preprocessed.xs, expected_xs, epsilon = 1e-6);
}

#[derive(Debug)]
pub struct LinearModel {
    pub intercept: f64,
    pub coefficients: Array1<f64>,
}

pub struct LinearRegression {
    pub fit_intercept: bool,
}

fn lstsq(
    xs: &Array2<f64>,
    ys: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let xs_f = xs.view().into_faer();
    // Compute the SVD of xs
    let svd = xs_f.svd().expect("SVD failed");
    let u = svd.U();
    // let s = svd.S();
    let vt = svd.V();

    // Compute the pseudo-inverse solution
    let s_recip = ys.clone(); // s.mapv(|v| if v > 1e-10 { 1.0 / v } else { 0.0 });
    let uh_y = u.into_ndarray().t().dot(ys);
    let coef = vt.into_ndarray().t().dot(&(&s_recip * &uh_y));

    // Calculate rank
    let rank = 3.0; // s.into_ndarray().iter().filter(|&&v| v > 1e-10).count() as f64;
    let rank_array = Array1::from_elem(1, rank);

    // Return the coefficients, residuals, rank, and singular values
    let residuals = ys - &xs_f.into_ndarray().dot(&coef);

    (coef, residuals, rank_array, ys.clone())
}

impl LinearRegression {
    pub fn fit(&self, xs: &Array2<f64>, ys: &Array1<f64>) -> LinearModel {
        let preprocessed = preprocess_data(xs, ys, self.fit_intercept);

        // coef, _, rank, singular = lstsq(xs, ys);
        // coef = np.ravel(coef)
        // intercept = set_intercept(preprocesssed.x_offset, preprocessed.y_offset);

        LinearModel {
            intercept: preprocessed.y_offset,
            coefficients: Array1::zeros(xs.shape()[1]),
        }
    }
}
