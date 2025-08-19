use crate::{GlyphMeta, MiniTypeFont};

/// Streaming iterator over a glyph's pixels (row-major).
/// Produces `u8` alpha values (already palette-expanded) for exactly
/// `glyph.w * font.atlas_height` pixels.
pub struct GlyphIter<'a> {
  font: &'a MiniTypeFont<'a>,
  meta: GlyphMeta,
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
    GlyphIter {
      font,
      meta,
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
    } else {
      // Row has no payload; keep payload_cursor unchanged for this row.
    }
    true
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
        let idx = if self.use_high { (b >> 4) & 0x0F } else { b & 0x0F } as usize;
        let alpha = self.font.palette[idx];

        // Update nibble state.
        if self.use_high {
          // consumed high nibble; move to next byte
          self.row_byte_index += 1;
          // Guard against reading past the row. We only load next byte
          // if more pixels remain in the row.
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
        // Whole row is black (palette[0]).
        self.remaining_in_row -= 1;
        return Some(0);
      }
    }
  }
}
