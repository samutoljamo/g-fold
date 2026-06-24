/// Opt-in matrix-diff cross-check: Rust assembler vs CVXPY/Clarabel canonical form.
///
/// Run with:
///   cargo test -p gfold-core --test matrix_diff -- --ignored
///
/// Approach: permutation-tolerant comparison using sorted per-cone-type residual multisets
/// evaluated at 20 deterministic pseudo-random points.  Equivalent formulations yield
/// identical residual multisets for every point; a mismatch signals a genuine formulation
/// difference.
use gfold_core::assemble::assemble;
use gfold_core::config::Config;

// ------------------------------------------------------------------
// Minimal LCG — no external crate dependency.
// Parameters from Knuth / Numerical Recipes.
// ------------------------------------------------------------------
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*state >> 33) as f64) / ((1u64 << 31) as f64)
}

// ------------------------------------------------------------------
// Reconstruct a dense (nrows x ncols) matrix from COO data.
// ------------------------------------------------------------------
fn coo_to_dense(
    rows: &[u64],
    cols: &[u64],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; ncols]; nrows];
    for ((r, c), v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        m[*r as usize][*c as usize] += v;
    }
    m
}

// ------------------------------------------------------------------
// Compute b - A*x (residual) for a dense matrix representation.
// ------------------------------------------------------------------
fn residual_dense(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(row, &bi)| bi - row.iter().zip(x.iter()).map(|(a_ij, x_j)| a_ij * x_j).sum::<f64>())
        .collect()
}

// ------------------------------------------------------------------
// Compute b - A*x for the Rust CscMatrix representation.
// ------------------------------------------------------------------
fn residual_csc(
    colptr: &[usize],
    rowval: &[usize],
    nzval: &[f64],
    b: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let nrows = b.len();
    let mut ax = vec![0.0_f64; nrows];
    for col in 0..x.len() {
        let start = colptr[col];
        let end = colptr[col + 1];
        for k in start..end {
            ax[rowval[k]] += nzval[k] * x[col];
        }
    }
    b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect()
}

// ------------------------------------------------------------------
// Extract sorted residuals pooled by cone TYPE (z, nn, soc).
// Returns (zero_pool, nonneg_pool, soc_pool) each sorted ascending.
// ------------------------------------------------------------------
fn pool_residuals_by_type(
    residuals: &[f64],
    cones: &[(String, usize)], // (type, dim)
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut zero_pool = Vec::new();
    let mut nn_pool = Vec::new();
    let mut soc_pool = Vec::new();
    let mut row = 0usize;
    for (ctype, dim) in cones {
        let slice = &residuals[row..row + dim];
        match ctype.as_str() {
            "z" => zero_pool.extend_from_slice(slice),
            "nn" => nn_pool.extend_from_slice(slice),
            "soc" => soc_pool.extend_from_slice(slice),
            _ => {}
        }
        row += dim;
    }
    zero_pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    nn_pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    soc_pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (zero_pool, nn_pool, soc_pool)
}

// ------------------------------------------------------------------
// Convert Clarabel SupportedConeT to (type_str, dim) pairs.
// ------------------------------------------------------------------
fn rust_cone_list(prob: &gfold_core::assemble::Problem) -> Vec<(String, usize)> {
    use clarabel::solver::SupportedConeT;
    prob.cones
        .iter()
        .map(|c| match c {
            SupportedConeT::ZeroConeT(d) => ("z".to_string(), *d),
            SupportedConeT::NonnegativeConeT(d) => ("nn".to_string(), *d),
            SupportedConeT::SecondOrderConeT(d) => ("soc".to_string(), *d),
            _ => ("other".to_string(), 0),
        })
        .collect()
}

#[test]
#[ignore = "requires CVXPY export; run with: cargo test -p gfold-core --test matrix_diff -- --ignored"]
fn matrix_equivalence_default() {
    // ---- Load CVXPY export ----
    let export_json = std::fs::read_to_string("../gfold-fixtures/data/matrices_default.json")
        .expect("matrices_default.json not found — run: cd generator && python -m gfold --export-matrices ../rust/gfold-fixtures/data/matrices_default.json");
    let v: serde_json::Value = serde_json::from_str(&export_json).unwrap();

    let py_n_vars = v["n_vars"].as_u64().unwrap() as usize;
    let py_n_rows = v["n_rows"].as_u64().unwrap() as usize;

    // Parse CVXPY cone list
    let py_cones: Vec<(String, usize)> = v["cones"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["type"].as_str().unwrap().to_string(),
                c["dim"].as_u64().unwrap() as usize,
            )
        })
        .collect();

    // Parse COO matrix
    let coo_rows: Vec<u64> = v["A"]["rows"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let coo_cols: Vec<u64> = v["A"]["cols"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    let coo_vals: Vec<f64> = v["A"]["vals"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let py_b: Vec<f64> = v["b"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();

    let py_a_dense = coo_to_dense(&coo_rows, &coo_cols, &coo_vals, py_n_rows, py_n_vars);

    // ---- Assemble Rust problem ----
    let cfg = Config::default();
    let prob = assemble(&cfg);
    let rust_n_vars = prob.layout.nvars();
    let rust_n_rows = prob.b.len();
    let rust_cones = rust_cone_list(&prob);

    // ---- Cone-dimension multiset: build sorted Vec<(type, dim)> ----
    let mut rust_sorted_cones: Vec<(String, usize)> = rust_cones.clone();
    rust_sorted_cones.sort();
    let mut py_sorted_cones = py_cones.clone();
    py_sorted_cones.sort();

    // ---- (a) Variable and row count ----
    assert_eq!(
        rust_n_vars, py_n_vars,
        "FORMULATION MISMATCH: Rust n_vars={rust_n_vars} vs CVXPY n_vars={py_n_vars}. \
         CVXPY introduces {extra} extra auxiliary variables not present in the Rust assembly.",
        extra = (py_n_vars as isize - rust_n_vars as isize).abs()
    );
    assert_eq!(
        rust_n_rows, py_n_rows,
        "FORMULATION MISMATCH: Rust n_rows={rust_n_rows} vs CVXPY n_rows={py_n_rows}."
    );

    // ---- (b) Cone-dimension multiset ----
    assert_eq!(
        rust_sorted_cones, py_sorted_cones,
        "FORMULATION MISMATCH: Cone multisets differ.\n  Rust: {:?}\n  CVXPY: {:?}",
        rust_sorted_cones, py_sorted_cones
    );

    // ---- (c) Per-cone residual multiset comparison at 20 random points ----
    // Point lives in min(rust_n_vars, py_n_vars) dimensions; we use the smaller
    // space and pad with zeros for the other side (only reached if dims match above).
    let point_dim = rust_n_vars.min(py_n_vars);
    let mut state: u64 = 0xDEAD_BEEF_1234_5678;

    for trial in 0..20 {
        // Generate a random point in [−10, 10]^point_dim
        let x: Vec<f64> = (0..point_dim).map(|_| lcg(&mut state) * 20.0 - 10.0).collect();
        let x_rust = &x[..rust_n_vars];
        let x_py = &x[..py_n_vars];

        // Rust residual via CSC matrix-vector product
        let r_rust = residual_csc(
            &prob.a_mat.colptr,
            &prob.a_mat.rowval,
            &prob.a_mat.nzval,
            &prob.b,
            x_rust,
        );

        // CVXPY residual via dense matrix-vector product
        let r_py = residual_dense(&py_a_dense, &py_b, x_py);

        // Pool and sort by cone type
        let (rz_rust, rnn_rust, rsoc_rust) = pool_residuals_by_type(&r_rust, &rust_cones);
        let (rz_py, rnn_py, rsoc_py) = pool_residuals_by_type(&r_py, &py_cones);

        // Compare pools
        assert_eq!(rz_rust.len(), rz_py.len(),
            "trial {trial}: zero-cone pool size mismatch: Rust {} vs CVXPY {}", rz_rust.len(), rz_py.len());
        assert_eq!(rnn_rust.len(), rnn_py.len(),
            "trial {trial}: nonneg-cone pool size mismatch: Rust {} vs CVXPY {}", rnn_rust.len(), rnn_py.len());
        assert_eq!(rsoc_rust.len(), rsoc_py.len(),
            "trial {trial}: SOC pool size mismatch: Rust {} vs CVXPY {}", rsoc_rust.len(), rsoc_py.len());

        for (i, (a, b)) in rz_rust.iter().zip(rz_py.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-9,
                "trial {trial}: zero-cone pool[{i}] mismatch: Rust={a:.6e} CVXPY={b:.6e} diff={diff:.2e}");
        }
        for (i, (a, b)) in rnn_rust.iter().zip(rnn_py.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-9,
                "trial {trial}: nonneg-cone pool[{i}] mismatch: Rust={a:.6e} CVXPY={b:.6e} diff={diff:.2e}");
        }
        for (i, (a, b)) in rsoc_rust.iter().zip(rsoc_py.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-9,
                "trial {trial}: SOC pool[{i}] mismatch: Rust={a:.6e} CVXPY={b:.6e} diff={diff:.2e}");
        }
    }
}
