//! # MiniType Atlas: MCU-Friendly L4 + Palette with Row Skipping
//!
//! This module implements a compact, MCU-friendly grayscale atlas format for 8-bit L8 images using a 16-color palette, 4bpp packed indices, and row skipping.
//!
//! ## Format Summary
//! - **Palette**: 16 bytes. `palette[0]` is always 0x00 (black), `palette[15]` is 0xFF (white). The remaining 14 entries form a **monotonic, evenly spaced ramp in linear light** (perceptual) and are converted back to sRGB. If the image uses ≤15 distinct nonzero values, those exact values are preserved instead of optimizing.
//! - **Indices**: Each pixel is mapped to the nearest palette entry **in linear light** (ties resolved by lower index).
//! - **Packing**: Pixels stored as 4bpp indices, 2 per byte. Low nibble = left pixel, high nibble = right pixel. For odd width, final byte’s high nibble is 0 and must be ignored.
//! - **Row Mask**: A bitset of ceil(height/8) bytes, LSB-first. Bit y=1 → row data present; y=0 → row omitted (implicitly black, index 0).
//! - **Header Layout** (little-endian):
//!   - 0..2: width (u16 LE)
//!   - 2..4: height (u16 LE)
//!   - 4..20: palette[16] (u8)
//!   - 20..(20 + ceil(h/8)): row mask
//!   - ...: row payloads (for rows with mask bit=1)
//!
//! ## Encoding/Decoding
//! - **Encoding**: Build the palette, map each pixel to a palette index, construct the row mask, and emit the header followed by packed row payloads.
//!   Linearize (γ≈2.2) → map to the ramp by nearest level in **linear light**; optionally apply low-strength, edge‑aware error diffusion (Floyd–Steinberg) to hide banding.
//!   Use `compress_with_opts(..., opts)` to tune dithering/edge threshold.
//! - **Decoding**: Parse the header, palette, and row mask; for each row, if mask=0, fill row with palette[0], else read packed indices and expand to palette values.
//!
//! ## MCU Considerations
//! - Only byte/nibble operations; no bit-level parsing.
//! - Rows with only black are skipped via row mask.
//!
//! ## Example
//! ```
//! # use minitype::atlas::{compress, decompress};
//! let w: u16 = 64; let h: u16 = 8;
//! let img = vec![0xFF; (w as usize * h as usize)];
//! let blob = compress(&img, w, h);
//! let (dw, dh, out) = decompress(&blob).unwrap();
//! assert_eq!((dw, dh), (w, h));
//! assert_eq!(out, img);
//! ```
/// Options controlling quantization and dithering.
#[derive(Copy, Clone, Debug)]
pub struct QuantOptions {
  /// Enable edge-aware Floyd–Steinberg dithering (applied only near edges).
  pub dither: bool,
  /// Edge detection threshold in sRGB (0..=255). Larger = fewer pixels considered edges.
  pub edge_thresh: u8,
}

/// Sensible defaults: dithering on, moderate edge threshold.
pub const DEFAULT_OPTS: QuantOptions = QuantOptions { dither: true, edge_thresh: 12 };

pub fn compress_with_opts(pixels: &[u8], w: u16, h: u16, opts: QuantOptions) -> Vec<u8> {
  let (palette, map_idx) = build_palette_u8(pixels);
  let w_usize = w as usize;
  let h_usize = h as usize;
  let rowmask_len = (h_usize + 7) / 8;

  // Optional edge-aware Floyd–Steinberg dithering (linear light).
  let dither = opts.dither;
  let edge_thresh: u8 = opts.edge_thresh; // apply diffusion only near visible edges
  let mut err_curr = vec![0f32; w_usize];
  let mut err_next = vec![0f32; w_usize];
  let centers_lin = palette_centers_linear(&palette);

  let mut rowmask = vec![0u8; rowmask_len];
  let mut rows_data: Vec<Vec<u8>> = Vec::with_capacity(h_usize);

  for row in 0..h_usize {
    // Reset next-row errors at the start of each row
    err_next.fill(0.0);

    let row_start = row * w_usize;
    let mut has_nonzero = false;
    let mut indices = Vec::with_capacity((w_usize + 1) / 2);
    let mut i = 0;
    while i < w_usize {
      // --- Left pixel ---
      let x0 = i;
      let s0 = pixels[row_start + x0];
      // Edge detection: compare to left/up neighbors in sRGB space
      let mut edge0 = false;
      if x0 > 0 {
        edge0 |= s0.abs_diff(pixels[row_start + x0 - 1]) >= edge_thresh;
      }
      if row > 0 {
        edge0 |= s0.abs_diff(pixels[row_start - w_usize + x0]) >= edge_thresh;
      }
      // Linearize + add error (if dithering)
      let mut v0 = to_lin(s0);
      if dither && edge0 {
        v0 = (v0 + err_curr[x0]).clamp(0.0, 1.0);
      }
      // Quantize by nearest center in linear light
      let mut idx_l = 0u8;
      let mut bd = f32::INFINITY;
      for c in 0..16 {
        let d = (v0 - centers_lin[c]).abs();
        if d < bd {
          bd = d;
          idx_l = c as u8;
        }
      }
      if dither && edge0 {
        let q0 = centers_lin[idx_l as usize];
        let e = v0 - q0;
        // Distribute FS errors (7/16, 3/16, 5/16, 1/16)
        if x0 + 1 < w_usize {
          err_curr[x0 + 1] += e * (7.0 / 16.0);
        }
        if x0 > 0 {
          err_next[x0 - 1] += e * (3.0 / 16.0);
        }
        err_next[x0] += e * (5.0 / 16.0);
        if x0 + 1 < w_usize {
          err_next[x0 + 1] += e * (1.0 / 16.0);
        }
      }

      // --- Right pixel (if any) ---
      let mut idx_r = 0u8;
      if x0 + 1 < w_usize {
        let x1 = x0 + 1;
        let s1 = pixels[row_start + x1];
        let mut edge1 = false;
        edge1 |= s1.abs_diff(pixels[row_start + x0]) >= edge_thresh;
        if row > 0 {
          edge1 |= s1.abs_diff(pixels[row_start - w_usize + x1]) >= edge_thresh;
        }
        let mut v1 = to_lin(s1);
        if dither && edge1 {
          v1 = (v1 + err_curr[x1]).clamp(0.0, 1.0);
        }
        let mut best_i = 0u8;
        let mut bd1 = f32::INFINITY;
        for c in 0..16 {
          let d = (v1 - centers_lin[c]).abs();
          if d < bd1 {
            bd1 = d;
            best_i = c as u8;
          }
        }
        idx_r = best_i;
        if dither && edge1 {
          let q1 = centers_lin[idx_r as usize];
          let e = v1 - q1;
          if x1 + 1 < w_usize {
            err_curr[x1 + 1] += e * (7.0 / 16.0);
          }
          if x1 > 0 {
            err_next[x1 - 1] += e * (3.0 / 16.0);
          }
          err_next[x1] += e * (5.0 / 16.0);
          if x1 + 1 < w_usize {
            err_next[x1 + 1] += e * (1.0 / 16.0);
          }
        }
      } else {
        idx_r = 0;
      }

      if idx_l != 0 || idx_r != 0 {
        has_nonzero = true;
      }
      let packed = (idx_r << 4) | (idx_l & 0x0F);
      indices.push(packed);
      i += 2;
    }

    if dither {
      std::mem::swap(&mut err_curr, &mut err_next);
    }

    if has_nonzero {
      let mask_byte = row / 8;
      let mask_bit = row % 8;
      rowmask[mask_byte] |= 1 << mask_bit;
      rows_data.push(indices);
    } else {
      rows_data.push(Vec::new());
    }
  }

  let mut out = Vec::with_capacity(20 + rowmask_len + h_usize * ((w_usize + 1) / 2));
  out.extend_from_slice(&w.to_le_bytes());
  out.extend_from_slice(&h.to_le_bytes());
  out.extend_from_slice(&palette);
  out.extend_from_slice(&rowmask);

  for row in 0..h_usize {
    let mask_byte = row / 8;
    let mask_bit = row % 8;
    if (rowmask[mask_byte] & (1 << mask_bit)) != 0 {
      out.extend_from_slice(&rows_data[row]);
    }
  }

  out
}

/// Backward-compatible entrypoint using DEFAULT_OPTS.
pub fn compress(pixels: &[u8], w: u16, h: u16) -> Vec<u8> {
  compress_with_opts(pixels, w, h, DEFAULT_OPTS)
}

/// Decompress a simple L4+palette bytestream with row skipping into `(width, height, L8 pixels)`.
///
/// Parses the header, palette, and row mask. For each row:
/// - If row mask bit is 0: fill with palette[0] (black).
/// - If row mask bit is 1: read `ceil(width/2)` bytes, expand nibbles (LSB: left pixel) to palette values.
///
/// Returns error if the file is too small or ends prematurely.
#[allow(dead_code)]
pub fn decompress(data: &[u8]) -> Result<(u16, u16, Vec<u8>), anyhow::Error> {
  use anyhow::anyhow;
  if data.len() < 20 {
    return Err(anyhow!("file too small"));
  }
  let w = u16::from_le_bytes([data[0], data[1]]);
  let h = u16::from_le_bytes([data[2], data[3]]);
  let w_usize = w as usize;
  let h_usize = h as usize;
  let palette_bytes = &data[4..20];
  let mut palette = [0u8; 16];
  palette.copy_from_slice(palette_bytes);
  let rowmask_len = (h_usize + 7) / 8;
  if data.len() < 20 + rowmask_len {
    return Err(anyhow!("file too small"));
  }
  let rowmask = &data[20..20 + rowmask_len];
  let mut out = vec![0u8; w_usize * h_usize];
  let mut pos = 20 + rowmask_len;
  for row in 0..h_usize {
    let mask_byte = row / 8;
    let mask_bit = row % 8;
    let mask = (rowmask[mask_byte] >> mask_bit) & 1;
    let row_start = row * w_usize;
    if mask == 0 {
      // All black row
      for x in 0..w_usize {
        out[row_start + x] = palette[0];
      }
    } else {
      let bytes_needed = (w_usize + 1) / 2;
      if pos + bytes_needed > data.len() {
        return Err(anyhow!("eof"));
      }
      for i in 0..bytes_needed {
        let b = data[pos + i];
        let idx_l = (b & 0x0F) as usize;
        let idx_r = (b >> 4) as usize;
        let x = i * 2;
        if x < w_usize {
          if idx_l >= 16 {
            return Err(anyhow!("palette idx out of range"));
          }
          out[row_start + x] = palette[idx_l];
        }
        if x + 1 < w_usize {
          if idx_r >= 16 {
            return Err(anyhow!("palette idx out of range"));
          }
          out[row_start + x + 1] = palette[idx_r];
        }
      }
      pos += bytes_needed;
    }
  }
  Ok((w, h, out))
}

/// Build a 16-entry u8 palette as a **monotonic ramp in linear light** and a 256-entry
/// map from value -> nearest palette index (nearest measured in linear light).
///
/// Fast path: if there are ≤15 distinct nonzero values in the image, preserve them exactly.
pub fn build_palette_u8(pixels: &[u8]) -> ([u8; 16], [u8; 256]) {
  // Histogram & unique set
  let mut used = [false; 256];
  for &v in pixels {
    used[v as usize] = true;
  }

  // Fast path: preserve exact values when small palette suffices
  let mut uniques: Vec<u8> = (1u16..=255).filter(|&v| used[v as usize]).map(|v| v as u8).collect();
  if uniques.len() <= 15 {
    uniques.sort_unstable();
    let mut palette = [0u8; 16];
    palette[0] = 0;
    for (i, &v) in uniques.iter().take(15).enumerate() {
      palette[i + 1] = v;
    }
    // Build nearest-index map by linear-light distance (ties -> lower index)
    let centers = palette_centers_linear(&palette);
    let mut map_idx = [0u8; 256];
    for v in 0..=255 {
      let lv = to_lin(v as u8);
      let mut best_i = 0u8;
      let mut bd = f32::INFINITY;
      for i in 0..16 {
        let d = (lv - centers[i]).abs();
        if d < bd || (d == bd && (i as u8) < best_i) {
          bd = d;
          best_i = i as u8;
        }
      }
      map_idx[v as usize] = best_i;
    }
    return (palette, map_idx);
  }

  // Monotonic, evenly spaced ramp in linear light (perceptual), pinned at 0 and 1.
  let mut centers = [0f32; 16];
  centers[0] = 0.0;
  centers[15] = 1.0;
  for i in 1..15 {
    centers[i] = (i as f32) / 15.0;
  }
  // Convert to sRGB palette with minimal separation to keep strict monotonicity in byte space.
  let mut palette = [0u8; 16];
  for i in 0..16 {
    palette[i] = to_srgb(centers[i]);
  }
  palette[0] = 0;
  palette[15] = 255;
  // Enforce a small minimum separation (>=4) in sRGB to avoid accidental reversals after rounding.
  for i in 1..16 {
    let min_next = palette[i - 1].saturating_add(4).min(255);
    if palette[i] < min_next {
      palette[i] = min_next;
    }
  }
  // Build nearest-index map by **linear** distance
  let mut map_idx = [0u8; 256];
  for v in 0..=255 {
    let lv = to_lin(v as u8);
    let mut best_i = 0u8;
    let mut bd = f32::INFINITY;
    for i in 0..16 {
      let d = (lv - centers[i]).abs();
      if d < bd || (d == bd && (i as u8) < best_i) {
        bd = d;
        best_i = i as u8;
      }
    }
    map_idx[v] = best_i;
  }
  (palette, map_idx)
}

const GAMMA: f32 = 2.2;
#[inline]
fn to_lin(v: u8) -> f32 {
  ((v as f32) / 255.0).powf(GAMMA)
}
#[inline]
fn to_srgb(l: f32) -> u8 {
  (255.0 * l.clamp(0.0, 1.0).powf(1.0 / GAMMA)).round() as u8
}
#[inline]
fn palette_centers_linear(palette: &[u8; 16]) -> [f32; 16] {
  let mut c = [0.0f32; 16];
  for i in 0..16 {
    c[i] = to_lin(palette[i]);
  }
  c
}

#[cfg(test)]
mod tests {
  use super::{compress, decompress};

  #[inline]
  fn a4(n: u8) -> u8 {
    (n << 4) | n
  }

  fn roundtrip(w: u16, h: u16, f: impl Fn(usize) -> u8) {
    let n = (w as usize) * (h as usize);
    let mut px = vec![0u8; n];
    for i in 0..n {
      px[i] = f(i);
    }
    let blob = compress(&px, w, h);
    let (dw, dh, out) = decompress(&blob).expect("decode");
    assert_eq!((dw, dh), (w, h));
    assert_eq!(out, px);
  }

  #[test]
  fn t_all_white() {
    roundtrip(64, 8, |_| a4(15));
  }

  #[test]
  fn t_all_black() {
    roundtrip(16, 16, |_| a4(0));
  }

  #[test]
  fn t_checker() {
    let w = 37;
    let h = 11;
    roundtrip(w, h, |i| {
      let x = i % (w as usize);
      let y = i / (w as usize);
      a4(if (x + y) % 2 == 0 { 2 } else { 12 })
    });
  }

  #[test]
  fn t_long_run_cross_rows() {
    let w = 13;
    let h = 9;
    let n = (w as usize) * (h as usize);
    roundtrip(w, h, |i| if i < n / 3 * 2 { a4(1) } else { a4(9) });
  }

  #[test]
  fn t_palette_majority() {
    let w = 40;
    let h = 3;
    let n = (w as usize) * (h as usize);
    let mut px = vec![a4(3); n];
    for i in 0..(n * 3 / 4) {
      px[i] = a4(7);
    }
    let blob = compress(&px, w, h);
    let (_, _, out) = decompress(&blob).unwrap();
    assert_eq!(out, px);
  }

  #[test]
  fn t_row_skip_all_black_rows() {
    // Interleave black and non-black rows
    let w = 8;
    let h = 10;
    let mut px = vec![0u8; w as usize * h as usize];
    for row in 0..h {
      for col in 0..w {
        let idx = row as usize * w as usize + col;
        if row % 2 == 1 {
          px[idx] = 0xF0; // nonzero value
        }
      }
    }
    let blob = compress(&px, w as u16, h);
    let (dw, dh, out) = decompress(&blob).unwrap();
    assert_eq!((dw, dh), (w as u16, h));
    assert_eq!(out, px);
  }
}
