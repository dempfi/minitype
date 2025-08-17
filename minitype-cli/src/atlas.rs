//! # MiniType Atlas: MCU-Friendly L4 + Palette with Row Skipping
//!
//! This module implements a compact, MCU-friendly grayscale atlas format for 8-bit L8 images using a 16-color palette, 4bpp packed indices, and row skipping.
//!
//! ## Format Summary
//! - **Palette**: 16 bytes. `palette[0]` is always 0x00 (black), `palette[15]` is 0xFF (white). The remaining 14 entries are computed by a Lloyd–Max (1‑D k‑means) optimizer in **linear light** and then converted back to sRGB. If the image uses ≤15 distinct nonzero values, those exact values are preserved instead of optimizing.
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
pub fn compress(pixels: &[u8], w: u16, h: u16) -> Vec<u8> {
  let (palette, map_idx) = build_palette_u8(pixels);
  let w_usize = w as usize;
  let h_usize = h as usize;
  let rowmask_len = (h_usize + 7) / 8;
  let mut rowmask = vec![0u8; rowmask_len];
  let mut rows_data: Vec<Vec<u8>> = Vec::with_capacity(h_usize);

  for row in 0..h_usize {
    let row_start = row * w_usize;
    let mut has_nonzero = false;
    let mut indices = Vec::with_capacity((w_usize + 1) / 2);
    let mut i = 0;
    while i < w_usize {
      let idx_l = map_idx[pixels[row_start + i] as usize];
      if idx_l != 0 {
        has_nonzero = true;
      }
      let idx_r = if i + 1 < w_usize {
        let idx = map_idx[pixels[row_start + i + 1] as usize];
        if idx != 0 {
          has_nonzero = true;
        }
        idx
      } else {
        0u8
      };
      let packed = (idx_r << 4) | (idx_l & 0x0F);
      indices.push(packed);
      i += 2;
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

/// Decompress a simple L4+palette bytestream with row skipping into `(width, height, L8 pixels)`.
///
/// Parses the header, palette, and row mask. For each row:
/// - If row mask bit is 0: fill with palette[0] (black).
/// - If row mask bit is 1: read `ceil(width/2)` bytes, expand nibbles (LSB: left pixel) to palette values.
///
/// Returns error if the file is too small or ends prematurely.
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

/// Build a 16-entry u8 palette for the new format and a 256-entry map from value -> nearest palette index.
///
/// - Computes a 256-bin histogram.
/// - palette[0] = 0.
/// - palette[15] = 255.
/// - palette[1..15]: 14 entries computed by Lloyd–Max clustering in linear light or preserved exact values if ≤15 distinct nonzero.
/// - Returns a palette and a value->nearest-index LUT (ties by lower index).
pub fn build_palette_u8(pixels: &[u8]) -> ([u8; 16], [u8; 256]) {
  // Histogram & unique set
  let mut hist = [0usize; 256];
  let mut used = [false; 256];
  for &v in pixels {
    hist[v as usize] += 1;
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
    // Build nearest-index map by sRGB distance (exact matches will map back 1:1)
    let mut map_idx = [0u8; 256];
    for v in 0u16..=255 {
      // ties -> lower index
      let mut best_i = 0u8;
      let mut best_d = u16::MAX;
      for (i, &p) in palette.iter().enumerate() {
        let d = if v as i32 >= p as i32 {
          v as i32 - p as i32
        } else {
          p as i32 - v as i32
        } as u16;
        if d < best_d || (d == best_d && (i as u8) < best_i) {
          best_d = d;
          best_i = i as u8;
        }
      }
      map_idx[v as usize] = best_i;
    }
    return (palette, map_idx);
  }

  // Lloyd–Max in linear light for 16 levels with 0 and 255 pinned
  const GAMMA: f32 = 2.2;
  #[inline]
  fn to_lin(v: u8) -> f32 {
    ((v as f32) / 255.0).powf(GAMMA)
  }

  #[inline]
  fn to_srgb(l: f32) -> u8 {
    (255.0 * l.clamp(0.0, 1.0).powf(1.0 / GAMMA)).round() as u8
  }

  // Precompute linear positions and weights per bin
  let mut lin = [0f32; 256];
  for v in 0..=255 {
    lin[v] = to_lin(v as u8);
  }

  // Initialize centers uniformly in linear space (with pins at 0 and 1)
  let k = 16usize; // total
  let free = 14usize; // free centers between 0 and 1
  let mut centers = [0f32; 16];
  centers[0] = 0.0;
  centers[15] = 1.0;
  for i in 0..free {
    centers[i + 1] = (i as f32 + 1.0) / (free as f32 + 1.0);
  }

  // Lloyd–Max iterations
  for _ in 0..16 {
    // Assign each bin to nearest center (linear distance)
    let mut sum = [0f64; 16];
    let mut wsum = [0u64; 16];
    for v in 0..=255 {
      let w = hist[v] as u64;
      if w == 0 {
        continue;
      }
      // evaluate distances to all centers; 16 small, fine
      let lv = lin[v];
      let mut best = 0usize;
      let mut bd = f32::INFINITY;
      for i in 0..k {
        let d = (lv - centers[i]).abs();
        if d < bd {
          bd = d;
          best = i;
        }
      }
      sum[best] += (lv as f64) * (w as f64);
      wsum[best] += w;
    }
    // Recompute free centers as weighted means; keep 0 & 1 pinned
    for i in 1..15 {
      if wsum[i] > 0 {
        centers[i] = (sum[i] / wsum[i] as f64) as f32;
      }
    }
    // Enforce monotonicity
    for i in 1..15 {
      if centers[i] < centers[i - 1] {
        centers[i] = centers[i - 1];
      }
    }
    for i in (1..15).rev() {
      if centers[i] > centers[i + 1] {
        centers[i] = centers[i + 1];
      }
    }
  }

  // Convert centers to sRGB palette and enforce a minimum separation in sRGB
  let mut palette = [0u8; 16];
  for i in 0..16 {
    palette[i] = to_srgb(centers[i]);
  }
  palette[0] = 0;
  palette[15] = 255;
  // Minimum separation of 4 in sRGB
  for i in 1..16 {
    if palette[i] < palette[i - 1].saturating_add(4) {
      palette[i] = palette[i - 1].saturating_add(4).min(255);
    }
  }

  // Build nearest-index map by **linear** distance
  let mut map_idx = [0u8; 256];
  for v in 0..=255 {
    let lv = lin[v];
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
