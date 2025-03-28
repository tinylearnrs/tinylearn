//! This is a copy of the `bfgs` crate. The crate can't be used since it's
//! locked to a too old version of `ndarray`.
//!
//! This package contains an implementation of
//! [BFGS](https://en.wikipedia.org/w/index.php?title=BFGS_method), an algorithm for minimizing
//! convex twice-differentiable functions.
//!
//! BFGS is explained at a high level in
//! [the blog post](https://paulkernfeld.com/2018/08/06/rust-needs-bfgs.html) introducing this
//! package.
//!
//! In this example, we minimize a 2d function:
//!
//! ```rust
//! extern crate bfgs;
//! extern crate ndarray;
//!
//! use ndarray::{Array, Array1};
//!
//! fn main() {
//!     let x0 = Array::from_vec(vec![8.888, 1.234]);  // Chosen arbitrarily
//!     let f = |x: &Array1<f64>| x.dot(x);
//!     let g = |x: &Array1<f64>| 2.0 * x;
//!     let x_min = bfgs::bfgs(x0, f, g);
//!     assert_eq!(x_min, Ok(Array::from_vec(vec![0.0, 0.0])));
//! }
//! ```
//!
//! This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
//! run `cargo make all`.

use core::f64::INFINITY;
use ndarray::{Array1, Array2};

const F64_MACHINE_EPSILON: f64 = 2e-53;

// From the L-BFGS paper (Zhu et al. 1994), 1e7 is for "moderate accuracy." 1e12 for "low
// accuracy," 10 for "high accuracy." If factr is 0, the algorithm will only stop if the value of f
// stops improving completely.
const FACTR: f64 = 1e7;

// This is FTOL from Zhu et al.
const F_TOLERANCE: f64 = FACTR * F64_MACHINE_EPSILON;

// Dumbly try many values of epsilon, taking the best one
// Return the value of epsilon that minimizes f
fn line_search<F>(f: F) -> Result<f64, ()>
where
    F: Fn(f64) -> f64,
{
    let mut best_epsilon = 0.0;
    let mut best_val_f = INFINITY;

    for i in -20..20 {
        let epsilon = 2.0_f64.powi(i);
        let val_f = f(epsilon);
        if val_f < best_val_f {
            best_epsilon = epsilon;
            best_val_f = val_f;
        }
    }
    if best_epsilon == 0.0 {
        Err(())
    } else {
        Ok(best_epsilon)
    }
}

fn new_identity_matrix(len: usize) -> Array2<f64> {
    let mut result = Array2::zeros((len, len));
    for z in result.diag_mut() {
        *z = 1.0;
    }
    result
}

// If the improvement in f is not too much bigger than the rounding error, then call it a
// success. This is the first stopping criterion from Zhu et al.
fn stop(f_x_old: f64, f_x: f64) -> bool {
    let negative_delta_f = f_x_old - f_x;
    let denom = f_x_old.abs().max(f_x.abs()).max(1.0);
    negative_delta_f / denom <= F_TOLERANCE
}

/// Returns a value of `x` that should minimize `f`. `f` must be convex and twice-differentiable.
///
/// - `x0` is an initial guess for `x`. Often this is chosen randomly.
/// - `f` is the objective function
/// - `g` is the gradient of `f`
#[allow(clippy::many_single_char_names)]
pub(crate) fn bfgs<F, G>(x0: Array1<f64>, f: F, g: G) -> Result<Array1<f64>, ()>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x = x0;
    let mut f_x = f(&x);
    let mut g_x = g(&x);
    let p = x.len();
    assert_eq!(g_x.dim(), x.dim());

    // Initialize the inverse approximate Hessian to the identity matrix
    let mut b_inv = new_identity_matrix(x.len());

    loop {
        // Find the search direction
        let search_dir = -1.0 * b_inv.dot(&g_x);

        // Find a good step size
        let epsilon = line_search(|epsilon| f(&(&search_dir * epsilon + &x))).map_err(|_| ())?;

        // Save the old values
        let f_x_old = f_x;
        let g_x_old = g_x;

        // Take a step in the search direction
        x.scaled_add(epsilon, &search_dir);
        f_x = f(&x);
        g_x = g(&x);

        // Compute deltas between old and new
        let y: Array2<f64> = (&g_x - &g_x_old)
            .into_shape_with_order((p, 1))
            .expect("y into_shape failed");
        let s: Array2<f64> = (epsilon * search_dir)
            .into_shape_with_order((p, 1))
            .expect("s into_shape failed");
        let sy: f64 = s
            .t()
            .dot(&y)
            .into_shape_with_order(())
            .expect("sy into_shape failed")[()];
        let ss: Array2<f64> = s.dot(&s.t());

        if stop(f_x_old, f_x) {
            return Ok(x);
        }

        // Update the Hessian approximation
        let to_add: Array2<f64> = ss * (sy + &y.t().dot(&b_inv.dot(&y))) / sy.powi(2);
        let to_sub: Array2<f64> = (b_inv.dot(&y).dot(&s.t()) + s.dot(&y.t().dot(&b_inv))) / sy;
        b_inv = b_inv + to_add - to_sub;
    }
}
