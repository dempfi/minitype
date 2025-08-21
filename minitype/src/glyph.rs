use crate::ceil_div_u16;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlyphId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlyphMetrics {
  pub w: u8,        // u7
  pub h: u8,        // u7
  pub yoff: u8,     // u7 (line-top → first row)
  pub left: i8,     // i4
  pub advance: i16, // computed: w + adv_delta (i5)

  pub(crate) offset: u32, // u18 (byte offset in atlas blob)
}

impl GlyphMetrics {
  /// Decode 6-byte glyph record at absolute `off`.
  #[inline]
  pub(crate) fn decode(data: &[u8], offset: usize) -> Option<Self> {
    if offset + 6 > data.len() {
      return None;
    }
    let bits = (data[offset] as u64)
      | ((data[offset + 1] as u64) << 8)
      | ((data[offset + 2] as u64) << 16)
      | ((data[offset + 3] as u64) << 24)
      | ((data[offset + 4] as u64) << 32)
      | ((data[offset + 5] as u64) << 40);

    let w = ((bits >> 0) & 0x7F) as u8;
    let h = ((bits >> 7) & 0x7F) as u8;
    let yoff = ((bits >> 14) & 0x7F) as u8;
    let advd5 = ((bits >> 21) & 0x1F) as u8; // i5
    let left4 = ((bits >> 26) & 0x0F) as u8; // i4
    let off18 = ((bits >> 30) & 0x3FFFF) as u32; // u18

    let adv_delta = sign_extend_5(advd5) as i16;
    let advance = (w as i16) + adv_delta;
    let left = sign_extend_4(left4);

    Some(Self { w, h, yoff, left, offset: off18, advance })
  }
}

/// Streaming iterator over a glyph's pixels (row-major).
/// Emits `u8` alpha with linear L4 reconstruction (alpha = nibble << 4).
pub(crate) struct GlyphPixels<'a> {
  slab: &'a [u8], // contiguous rows for this glyph
  bpr: usize,     // bytes per row = ceil(w, 2)
  w: u16,         // pixels per row

  // iteration state
  remaining_total: usize, // total pixels left (w * rows)
  remaining_in_row: u16,  // pixels left in the current row
  row: u16,               // next row to initialize (0..rows)
  byte_index: usize,      // index into `slab` for current row
  use_high: bool,         // false -> low nibble next, true -> high nibble next
  cur: u8,                // current byte cache
}

impl<'a> GlyphPixels<'a> {
  #[inline(always)]
  pub(crate) fn new(slab: &'a [u8], metrics: GlyphMetrics) -> Self {
    Self {
      slab,
      bpr: ceil_div_u16(metrics.w as u16, 2) as usize,
      w: metrics.w as u16,
      remaining_total: (metrics.w as usize).saturating_mul(metrics.h as usize),
      remaining_in_row: 0,
      row: 0,
      byte_index: 0,
      use_high: false,
      cur: 0,
    }
  }
}

impl<'a> Iterator for GlyphPixels<'a> {
  type Item = u8; // 0..=255
  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    // No pixels left at all?
    if self.remaining_total == 0 {
      return None;
    }

    // Need to start a new row?
    if self.remaining_in_row == 0 {
      // Initialize row state. If w == 0 this block won't run because remaining_total == 0.
      let base = (self.row as usize) * self.bpr;
      self.row += 1;

      self.remaining_in_row = self.w;
      self.byte_index = base;
      self.use_high = false;
      // Safe because w>0 => bpr>0, and `slab` length is rows*bpr
      self.cur = unsafe { *self.slab.get_unchecked(self.byte_index) };
    }

    let shift = (self.use_high as u8 as u32) * 4;
    let nib = ((self.cur as u32) >> shift) as u8 & 0x0F;
    let alpha = nib << 4;

    // Bookkeeping
    self.remaining_total -= 1;
    self.remaining_in_row -= 1;

    if self.use_high {
      // Consumed the high nibble: advance to the next byte for the next pixel (if any).
      self.use_high = false;
      self.byte_index += 1;
      if self.remaining_in_row > 0 {
        self.cur = unsafe { *self.slab.get_unchecked(self.byte_index) };
      }
    } else {
      // Consumed the low nibble: if row continues, next is the high nibble of the same byte.
      if self.remaining_in_row > 0 {
        self.use_high = true;
      }
      // else: row ended on an odd width; do not flip to high, next iteration will init a new row.
    }

    Some(alpha)
  }
}

#[inline]
fn sign_extend_5(v: u8) -> i8 {
  ((v as i8) << 3) >> 3
}
#[inline]
fn sign_extend_4(v: u8) -> i8 {
  ((v as i8) << 4) >> 4
}
