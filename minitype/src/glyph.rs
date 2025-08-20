#![allow(dead_code)]

use crate::{GlyphMeta, MiniTypeFont};

/// Streaming iterator over a glyph's pixels (row-major).
/// Produces `u8` alpha values using a per-glyph fused LUT that combines:
///   - the selected palette (nonlinear per-font fit), and
///   - the bias/step hint (per-glyph dynamic).
///
/// Exact cost per pixel: one nibble extraction + one table lookup.
pub struct GlyphIter<'a> {
  font: &'a MiniTypeFont<'a>,
  meta: GlyphMeta,

  // Precomputed fused 16-entry alpha table for this glyph.
  fused_lut: [u8; 16],

  // iteration state
  y: u16,                // 0..font.atlas_height (tight)
  remaining_in_row: u16, // countdown over glyph width
  row_byte_index: usize, // row start + (x/2)
  use_high: bool,        // if true -> use high nibble of current byte
  cur_byte: u8,
}

impl<'a> GlyphIter<'a> {
  pub(crate) fn new(font: &'a MiniTypeFont<'a>, meta: GlyphMeta) -> Self {
    // bias in 0..240 (steps of 16); step from the 4-value table.
    let bias = meta.bias16 * 16u16;
    let step = [8u16, 12u16, 16u16, 24u16][meta.scale_idx];

    // Build fused LUT: mix palette + bias/step, then enforce monotonicity.
    let palette = &font.palettes()[meta.palette_id];
    let mut lut = [0u8; 16];
    lut[0] = 0;

    // Weighted blend: favor bias/step (captures glyph-specific dynamic) but
    // still respect palette curvature. fused = round((3*p + 5*t)/8).
    let mut prev = 0u16;
    let mut n = 1usize;
    while n < 16 {
      let p = palette[n] as u16;
      let t_raw = bias + step.saturating_mul(n as u16);
      let t = if t_raw > 255 { 255 } else { t_raw } as u16;
      let fused = ((3 * p + 5 * t + 4) >> 3) as u16;
      let m = if fused < prev { prev } else { fused };
      prev = m;
      lut[n] = m as u8;
      n += 1;
    }

    GlyphIter {
      font,
      meta,
      fused_lut: lut,
      y: 0,
      remaining_in_row: 0, // triggers row init on first `next()`
      row_byte_index: 0,
      use_high: false,
      cur_byte: 0,
    }
  }

  #[inline]
  pub fn width(&self) -> u16 {
    self.meta.w as u16
  }

  /// Height of the tight band serialized in the atlas.
  #[inline]
  pub fn height(&self) -> u16 {
    self.font.atlas_height
  }

  /// Vertical offset of the tight band within the original full-height atlas.
  /// Useful if your renderer needs absolute Y positioning.
  #[inline]
  pub fn y_offset(&self) -> u16 {
    self.font.atlas_y_off
  }

  // Initialize state for the next row, if any.
  #[inline]
  fn begin_row(&mut self) -> bool {
    if self.y >= self.font.atlas_height {
      return false;
    }
    self.remaining_in_row = self.meta.w as u16;

    // Compute row start in contiguous payload, then position at glyph.x.
    let row_start = self.font.payload_off + (self.y as usize) * self.font.row_stride_bytes;
    let x = self.meta.x as usize;
    self.row_byte_index = row_start + (x / 2);
    self.use_high = (x & 1) != 0; // odd x starts at the high nibble
    self.cur_byte = self.font.data[self.row_byte_index];
    true
  }

  #[inline]
  fn expand_nibble(&self, n: u8) -> u8 {
    // 0 stays 0 by construction of the LUT.
    self.fused_lut[n as usize]
  }
}

impl<'a> Iterator for GlyphIter<'a> {
  type Item = u8; // alpha

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      // Start a new row if needed.
      if self.remaining_in_row == 0 {
        if !self.begin_row() {
          return None;
        }
        self.y += 1; // row is now active
      }

      // Emit one pixel from the current row.
      let b = self.cur_byte;
      let nib = if self.use_high { (b >> 4) & 0x0F } else { b & 0x0F };

      // Reconstruct alpha via fused LUT.
      let alpha = self.expand_nibble(nib);

      // Update nibble/byte state.
      if self.use_high {
        // consumed high nibble; move to next byte
        self.row_byte_index += 1;
        if self.remaining_in_row > 1 {
          self.cur_byte = self.font.data[self.row_byte_index];
        }
        self.use_high = false;
      } else {
        // consumed low nibble; next is high of the same byte
        self.use_high = true;
      }

      self.remaining_in_row -= 1;
      return Some(alpha);
    }
  }
}
