use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use bytemuck::{cast_slice, try_cast_slice};
use image::{ImageBuffer, Rgba};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

static LOG_ENABLED: OnceLock<bool> = OnceLock::new();
static LOG_EVERY: OnceLock<u64> = OnceLock::new();
static SERIALIZE_CALLS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn log_enabled() -> bool {
  *LOG_ENABLED.get_or_init(|| {
    std::env::var("RSTRAINER_LOG")
      .ok()
      .map(|v| {
        let v = v.to_ascii_lowercase();
        !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
      })
      .unwrap_or(false)
  })
}

#[inline]
fn log_every() -> u64 {
  *LOG_EVERY.get_or_init(|| {
    std::env::var("RSTRAINER_LOG_EVERY")
      .ok()
      .and_then(|v| v.parse::<u64>().ok())
      .filter(|&n| n > 0)
      .unwrap_or(1000)
  })
}

#[inline]
fn log_serialize(rows: usize, cols: usize, vmax: f32, true_max: f32) {
  if !log_enabled() {
    return;
  }
  let n = SERIALIZE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
  if n % log_every() == 0 {
    eprintln!(
      "[rstrainer] serialize calls={} shape=({},{}) vmax={} true_max={}",
      n, rows, cols, vmax, true_max
    );
  }
}


#[inline]
fn clamp_u8_bin(v: f32) -> usize {
  if !v.is_finite() || v <= 0.0 {
    0
  } else if v >= 255.0 {
    255
  } else {
    v as usize
  }
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
  clamp_u8_bin(v) as u8
}

#[inline]
fn lut_rgba(i: u8) -> [u8; 4] {
  let g = 255u8.wrapping_sub(i);
  [255, g, g, 255]
}

#[inline]
fn min_and_denom(vals: &[f32], vmax: f32) -> Option<(f32, f32)> {
  if !vmax.is_finite() {
    return None;
  }
  let mut min_v = f32::INFINITY;
  for &v in vals {
    if !v.is_finite() {
      return None;
    }
    if v < min_v {
      min_v = v;
    }
  }
  if vmax <= min_v {
    None
  } else {
    Some((min_v, vmax - min_v))
  }
}


#[inline]
fn compute_hists<I: Iterator<Item = f32>>(
  it: I,
  vmax: f32,
  true_max: f32,
) -> ([u32; 256], [u32; 256]) {
  let mut rel = [0u32; 256];
  let mut tru = [0u32; 256];

  let vd = if vmax > 0.0 { vmax } else { 1.0 };
  let td = if true_max > 0.0 { true_max } else { 1.0 };

  for x in it {
    rel[clamp_u8_bin((x / vd) * 255.0)] += 1;
    tru[clamp_u8_bin((x / td) * 255.0)] += 1;
  }
  (rel, tru)
}

#[pyfunction]
fn histograms<'py>(
  py: Python<'py>,
  matrix: PyReadonlyArray2<'py, f32>,
  vmax: f32,
  true_max: f32,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
  let m = matrix.as_array();
  let (hr, ht) = compute_hists(m.iter().copied(), vmax, true_max);
  Ok((
    PyBytes::new_bound(py, cast_slice(&hr)),
    PyBytes::new_bound(py, cast_slice(&ht)),
  ))
}

#[pyfunction]
fn serialize_window_and_hists<'py>(
  py: Python<'py>,
  matrix: PyReadonlyArray2<'py, f32>,
  vmax: f32,
  true_max: f32,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
  let m = matrix.as_array();
  let (rows, cols) = (m.nrows(), m.ncols());
  log_serialize(rows, cols, vmax, true_max);

  let bytes = if let Some(s) = m.as_slice() {
    PyBytes::new_bound(py, cast_slice(s))
  } else {
    let owned = m.to_owned();
    PyBytes::new_bound(py, cast_slice(owned.as_slice().unwrap()))
  };

  let (hr, ht) = compute_hists(m.iter().copied(), vmax, true_max);
  Ok((
    bytes,
    PyBytes::new_bound(py, cast_slice(&hr)),
    PyBytes::new_bound(py, cast_slice(&ht)),
  ))
}


fn parse_f32_blob<'a>(
  blob: &'a [u8],
  dim_hint: usize,
) -> Result<(usize, &'a [f32]), &'static str> {
  let vals: &[f32] = try_cast_slice(blob).map_err(|_| "bad f32 blob")?;
  let n = vals.len();
  for d in [dim_hint, dim_hint + 1, dim_hint.saturating_sub(1)] {
    if d > 0 && d * d == n {
      return Ok((d, vals));
    }
  }
  Err("dim mismatch")
}


#[pyfunction]
fn make_png_from_f32_blob(
  py_blob: &Bound<'_, PyBytes>,
  dim_hint: usize,
  viewing_vmax: f32,
  out_path: &str,
) -> PyResult<(bool, usize)> {
  let (dim, vals) = match parse_f32_blob(py_blob.as_bytes(), dim_hint) {
    Ok(v) => v,
    Err(_) => return Ok((false, 0)),
  };
  let (min_v, denom) = match min_and_denom(vals, viewing_vmax) {
    Some(v) => v,
    None => return Ok((false, dim)),
  };

  let mut rgba = vec![0u8; dim * dim * 4];
  rgba.par_chunks_exact_mut(4)
    .zip(vals.par_iter())
    .for_each(|(px, &v)| {
      let idx = clamp_u8(((v - min_v) / denom) * 255.0);
      px.copy_from_slice(&lut_rgba(idx));
    });

  let img = ImageBuffer::<Rgba<u8>, _>::from_vec(dim as u32, dim as u32, rgba)
    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("bad image"))?;

  if let Some(p) = Path::new(out_path).parent() {
    if !p.as_os_str().is_empty() {
      std::fs::create_dir_all(p)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }
  }
  img.save(out_path)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
  Ok((true, dim))
}

#[pyfunction]
fn render_rgb_u8_chw_from_f32_blob<'py>(
  py: Python<'py>,
  py_blob: &Bound<'py, PyBytes>,
  dim_hint: usize,
  viewing_vmax: f32,
  out_h: usize,
  out_w: usize,
) -> PyResult<(Bound<'py, PyBytes>, usize)> {
  let (dim, vals) = match parse_f32_blob(py_blob.as_bytes(), dim_hint) {
    Ok(v) => v,
    Err(_) => return Ok((PyBytes::new_bound(py, b""), 0)),
  };
  let (min_v, denom) = match min_and_denom(vals, viewing_vmax) {
    Some(v) => v,
    None => return Ok((PyBytes::new_bound(py, b""), dim)),
  };

  let n = out_h * out_w;
  let mut out = vec![0u8; 3 * n];

  for oy in 0..out_h {
    let iy = (oy * dim) / out_h;
    for ox in 0..out_w {
      let ix = (ox * dim) / out_w;
      let v = vals[iy * dim + ix];
      let idx = clamp_u8(((v - min_v) / denom) * 255.0);
      let g = 255u8.wrapping_sub(idx);
      let p = oy * out_w + ox;
      out[p] = 255;
      out[n + p] = g;
      out[2 * n + p] = g;
    }
  }

  Ok((PyBytes::new_bound(py, &out), dim))
}


#[inline]
fn n_ratio_raw(seq: &str) -> f32 {
  if seq.is_empty() {
    return 1.0;
  }
  let n = seq
    .as_bytes()
    .iter()
    .filter(|&&b| b.to_ascii_uppercase() == b'N')
    .count();
  n as f32 / seq.len() as f32
}

fn clean_sequence_fast(seq: &str) -> String {
  let mut out = Vec::with_capacity(seq.len());
  let mut in_header = false;
  for &b in seq.as_bytes() {
    match b {
      b'>' => in_header = true,
      b'\n' | b'\r' if in_header => in_header = false,
      _ if in_header || b.is_ascii_whitespace() => {}
      _ => {
        let u = b.to_ascii_uppercase();
        if matches!(u, b'A' | b'C' | b'G' | b'T' | b'N') {
          out.push(u);
        }
      }
    }
  }
  unsafe { String::from_utf8_unchecked(out) }
}

#[pyfunction]
fn clean_pair_and_n_ratio(
  seq_a: &str,
  seq_b: Option<&str>,
) -> PyResult<(bool, String, String, f32, f32)> {
  let nra = n_ratio_raw(seq_a);
  let nrb = seq_b.map(n_ratio_raw).unwrap_or(0.0);
  Ok((
    true,
    clean_sequence_fast(seq_a),
    seq_b.map(clean_sequence_fast).unwrap_or_default(),
    nra,
    nrb,
  ))
}

#[pymodule]
fn rstrainer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(histograms, m)?)?;
  m.add_function(wrap_pyfunction!(serialize_window_and_hists, m)?)?;
  m.add_function(wrap_pyfunction!(make_png_from_f32_blob, m)?)?;
  m.add_function(wrap_pyfunction!(render_rgb_u8_chw_from_f32_blob, m)?)?;
  m.add_function(wrap_pyfunction!(clean_pair_and_n_ratio, m)?)?;
  Ok(())
}
