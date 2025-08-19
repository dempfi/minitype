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
  y: u16,
  remaining_in_row: u16,
  payload_cursor: usize, // global cursor across present rows
  // per-row decoding state (valid only while remaining_in_row > 0)
  row_present: bool,
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
    // still respect palette curvature.  fused = round((3*p + 5*t)/8).
    // This works well across light/dark glyphs without ringing.
    let mut prev = 0u16;
    let mut n = 1usize;
    while n < 16 {
      let p = palette[n] as u16;
      let t_raw = bias + step.saturating_mul(n as u16);
      let t = if t_raw > 255 { 255 } else { t_raw } as u16;

      // fused = round((3*p + 5*t)/8)  ( +4 for rounding )
      let fused = ((3 * p + 5 * t + 4) >> 3) as u16;

      // Monotonic forward pass (cheap isotonic approximation).
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
      payload_cursor: font.payload_off,
      row_present: false,
      row_byte_index: 0,
      use_high: false,
      cur_byte: 0,
    }
  }

  #[inline]
  pub fn width(&self) -> u16 {
    self.meta.w as u16
  }

  #[inline]
  pub fn height(&self) -> u16 {
    self.font.atlas_height
  }

  // Initialize state for the next row (if any).
  fn begin_row(&mut self) -> bool {
    if self.y >= self.font.atlas_height {
      return false;
    }
    self.row_present = self.font.row_present(self.y);
    self.remaining_in_row = self.meta.w as u16;

    if self.row_present {
      // Compute row start in payload (current payload_cursor), then
      // position at the glyph's x offset.
      let row_start = self.payload_cursor;
      let x = self.meta.x as usize;
      self.row_byte_index = row_start + (x / 2);
      self.use_high = (x & 1) != 0; // odd x starts at the high nibble
      self.cur_byte = self.font.data[self.row_byte_index];
    }
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
        if self.y >= self.font.atlas_height {
          return None;
        }
        // If we just finished a present row, advance payload_cursor by stride.
        if self.y > 0 {
          let prev_y = self.y - 1;
          if self.font.row_present(prev_y) {
            self.payload_cursor += self.font.row_stride_bytes;
          }
        }
        // Begin current row
        if !self.begin_row() {
          return None;
        }
        self.y += 1; // mark that row is active (y counts rows consumed)
      }

      // Emit one pixel from the current row.
      if self.row_present {
        let b = self.cur_byte;
        let nib = if self.use_high { (b >> 4) & 0x0F } else { b & 0x0F };

        // Reconstruct alpha via fused LUT.
        let alpha = self.expand_nibble(nib);

        // Update nibble state.
        if self.use_high {
          // consumed high nibble; move to next byte
          self.row_byte_index += 1;
          // Only load next byte if more pixels remain.
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
      } else {
        // Whole row is black.
        self.remaining_in_row -= 1;
        return Some(0);
      }
    }
  }
}
