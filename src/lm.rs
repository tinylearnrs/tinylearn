use faer_ext::*;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use faer::prelude::*;
use faer::linalg::solvers::SolveLstsq;
use faer::linalg::solvers::SolveLstsqCore;

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
        let xs_offset = Array1::zeros(xs.ncols());
        let y_offset = 0.0;
        PreprocessedData {
            xs,
            ys,
            xs_offset,
            y_offset,
        }
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

pub struct LsqsqResult {
    pub coef: Array1<f64>,
    pub residuals: Array1<f64>,
}

fn lstsq(xs: &Array2<f64>, ys: &Array1<f64>) -> LsqsqResult {
    let xs_f = xs.view().into_faer();
    // Create a copy for QR decomposition
    let xs_qr = xs_f.cloned();
    // Create a mutable copy for the solution
    let solution = ys.clone().into_shape_clone((ys.dim(), 1));
    let solution = solution.expect("Failed to clone ys");
    let mut solution_f = solution.view().into_faer().to_owned();
    let solution_mut = solution_f.as_mut();
    
    // We don't have to add a 1s column because the data was already centered.
    
    // Perform QR decomposition and solve the least squares problem
    let qr = xs_qr.qr();
    let conj = faer::Conj::No;
    qr.solve_lstsq_in_place_with_conj(conj, solution_mut);
    // let x = xs_f.as_ref().subrows(0, xs.nrows());
    let coef = solution_f.subrows(0, xs.nrows());
    let coef = coef.into_ndarray();
    let coef = coef.to_owned();
    tracing::info!("coef: {:?}", &coef);
    let ys2 = ys.clone();
    let coef = Array1::from_iter(coef.into_iter());
    let residuals = ys2 - &xs_f.into_ndarray().dot(&coef);

    tracing::info!("xs_f: {:?}", &xs_f.into_ndarray());
    LsqsqResult {
        coef,
        residuals,
    }
}

impl LinearRegression {
    pub fn fit(&self, xs: &Array2<f64>, ys: &Array1<f64>) -> LinearModel {
        let preprocessed = preprocess_data(xs, ys, self.fit_intercept);

        let lsqsq_result = lstsq(&preprocessed.xs, &preprocessed.ys);
        let intercept = if self.fit_intercept {
            preprocessed.y_offset - preprocessed.xs_offset.dot(&lsqsq_result.coef)
        } else {
            0.0
        };

        LinearModel {
            intercept,
            coefficients: lsqsq_result.coef,
        }
    }
}
