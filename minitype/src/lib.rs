#![no_std]

//! Minimal no_std MiniType font reader & streaming glyph iterator.
//!
//! Implements the spec provided in the project README. The reader takes a
//! `&[u8]` and exposes:
//! - Fast glyph metadata lookup by glyph index
//! - Codepoint→glyph index mapping via compact charset segments
//! - An O(1) setup, streaming `GlyphIter` that walks only the target glyph
//!   rectangle across the atlas rows without decoding the whole atlas.
//!
//! This module is `no_std`-friendly (uses only `core`).

use core::convert::TryInto;

/// Parsing/validation errors
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Error {
  /// Wrong magic (expected "ZFNT").
  BadMagic,
  /// Unsupported version (expected 1).
  BadVersion(u8),
  /// Header/glyph-table alignment/length errors.
  Malformed,
  /// Declared total length does not match input length.
  LengthMismatch,
  /// Inner atlas header too short or inconsistent.
  AtlasHeader,
  /// Atlas palette[0] must be 0x00.
  Palette0,
  /// Atlas payload shorter than required by row mask / width.
  AtlasTooShort,
  /// Glyph table record out of bounds.
  GlyphOob,
  /// Glyph references x+w beyond atlas width.
  GlyphBounds,
  /// Kerning block declared outside the file bounds.
  KerningBounds,
}

/// Compact glyph metadata (4 bytes/glyph in the file).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlyphMeta {
  pub x: u16,
  pub w: u8,
  pub advance: i8,
}

/// Charset segment record (7 bytes each in the file).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct CharsetSeg {
  start_cp: u32, // u24
  len: u16,
  glyph_base: u16,
}

/// Optional kerning pair (7 bytes each in the file).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KerningPair {
  pub left: u32,  // u24
  pub right: u32, // u24
  pub adj: i8,
}

/// Parsed MiniType font view over the provided bytes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MiniTypeFont<'a> {
  data: &'a [u8],

  // Header metrics
  pub line_height: u16,
  pub ascent: i16,
  pub descent: i16,

  // Counts
  glyph_count: u16,
  charset_seg_count: u16,
  kerning_count: u32,

  // Offsets
  header_len: usize, // == 0x2C + 7 * charset_seg_count
  glyph_table_off: usize,
  glyph_table_len: usize,
  charset_segs_off: usize, // always 0x2C
  atlas_off: usize,
  atlas_len: usize,
  kerning_off: usize, // 0 means none

  // Atlas inner header
  pub atlas_width: u16,
  pub atlas_height: u16,
  palette: [u8; 16],
  row_mask_off: usize,
  row_mask_len: usize,     // == ceil(h/8)
  payload_off: usize,      // == atlas_off + 20 + row_mask_len
  row_stride_bytes: usize, // == ceil(width/2)
}

impl<'a> MiniTypeFont<'a> {
  /// Parse and validate a font from bytes
  pub fn new(data: &'a [u8]) -> Result<Self, Error> {
    // ---- Fixed header (44 bytes) ----
    if data.len() < 44 {
      return Err(Error::Malformed);
    }
    if &data[0..4] != b"MFNT" {
      return Err(Error::BadMagic);
    }
    let version = data[4];
    if version != 1 {
      return Err(Error::BadVersion(version));
    }
    let _flags = data[5];
    let line_height = le_u16(&data[6..8]);
    let ascent = le_i16(&data[8..10]);
    let descent = le_i16(&data[10..12]);
    let glyph_count = le_u16(&data[12..14]);
    let glyph_table_off = le_u32(&data[14..18]) as usize;
    let glyph_table_len = le_u32(&data[18..22]) as usize;
    let atlas_off = le_u32(&data[22..26]) as usize;
    let atlas_len = le_u32(&data[26..30]) as usize;
    let total_len = le_u32(&data[30..34]) as usize;
    let kerning_off = le_u32(&data[34..38]) as usize;
    let kerning_count = le_u32(&data[38..42]);
    let charset_seg_count = le_u16(&data[42..44]);

    if total_len != data.len() {
      return Err(Error::LengthMismatch);
    }

    let charset_segs_off = 0x2C;
    let header_len = 0x2C + 7usize * (charset_seg_count as usize);
    if glyph_table_off != header_len {
      return Err(Error::Malformed);
    }
    if glyph_table_len != 4usize * (glyph_count as usize) {
      return Err(Error::Malformed);
    }
    if atlas_off != glyph_table_off + glyph_table_len {
      return Err(Error::Malformed);
    }
    if atlas_off + 20 > total_len {
      return Err(Error::AtlasHeader);
    }

    // ---- Inner atlas header ----
    let width = le_u16(&data[atlas_off..atlas_off + 2]);
    let height = le_u16(&data[atlas_off + 2..atlas_off + 4]);
    let palette_slice = &data[atlas_off + 4..atlas_off + 20];
    let mut palette = [0u8; 16];
    palette.copy_from_slice(palette_slice);
    if palette[0] != 0 {
      return Err(Error::Palette0);
    }

    let row_mask_len = ceil_div_u16(height, 8) as usize;
    let row_mask_off = atlas_off + 20;
    let payload_off = row_mask_off + row_mask_len;
    if payload_off > total_len {
      return Err(Error::AtlasHeader);
    }

    let row_stride_bytes = ceil_div_u16(width, 2) as usize;

    // Validate we have enough payload bytes for present rows.
    if atlas_off + atlas_len > total_len {
      return Err(Error::AtlasTooShort);
    }
    let present_rows = count_present_rows(&data[row_mask_off..row_mask_off + row_mask_len], height);
    let need_payload = present_rows as usize * row_stride_bytes;
    if atlas_len < 20 + row_mask_len + need_payload {
      return Err(Error::AtlasTooShort);
    }

    // Validate kerning block bounds if present.
    if kerning_off != 0 {
      let need = kerning_off + 7usize * (kerning_count as usize);
      if need > total_len || kerning_off < atlas_off + atlas_len {
        return Err(Error::KerningBounds);
      }
    }

    // Validate glyph table within file.
    if glyph_table_off + glyph_table_len > total_len {
      return Err(Error::Malformed);
    }

    // Validate each glyph fits within atlas width.
    for gi in 0..(glyph_count as usize) {
      let off = glyph_table_off + gi * 4;
      let x = le_u16(&data[off..off + 2]);
      let w = data[off + 2];
      let xw = x as u32 + w as u32;
      if xw > width as u32 {
        return Err(Error::GlyphBounds);
      }
    }

    Ok(Self {
      data,
      line_height,
      ascent,
      descent,
      glyph_count,
      charset_seg_count,
      kerning_count,
      header_len,
      glyph_table_off,
      glyph_table_len,
      charset_segs_off,
      atlas_off,
      atlas_len,
      kerning_off,
      atlas_width: width,
      atlas_height: height,
      palette,
      row_mask_off,
      row_mask_len,
      payload_off,
      row_stride_bytes,
    })
  }

  /// Number of glyphs.
  #[inline]
  pub fn glyph_count(&self) -> u16 {
    self.glyph_count
  }

  /// Fetch glyph metadata by glyph index.
  #[inline]
  pub fn glyph_meta(&self, index: u16) -> Option<GlyphMeta> {
    if index as usize >= self.glyph_count as usize {
      return None;
    }
    let off = self.glyph_table_off + (index as usize) * 4;
    if off + 4 > self.data.len() {
      return None;
    }
    let x = le_u16(&self.data[off..off + 2]);
    let w = self.data[off + 2];
    let advance = self.data[off + 3] as i8;
    Some(GlyphMeta { x, w, advance })
  }

  /// Map Unicode scalar value to glyph index using charset segments.
  /// Returns `None` if the codepoint is not covered.
  pub fn glyph_index_for_cp(&self, cp: u32) -> Option<u16> {
    let mut off = self.charset_segs_off;
    for _ in 0..self.charset_seg_count {
      if off + 7 > self.data.len() {
        return None;
      }
      let start = le_u24(&self.data[off..off + 3]);
      let len = le_u16(&self.data[off + 3..off + 5]);
      let base = le_u16(&self.data[off + 5..off + 7]);
      off += 7;

      if cp >= start && cp < start + len as u32 {
        let idx = base as u32 + (cp - start);
        if idx < self.glyph_count as u32 {
          return Some(idx as u16);
        } else {
          return None;
        }
      }
    }
    None
  }

  /// Optional kerning lookup. Returns adjustment if found.
  pub fn kerning(&self, left_cp: u32, right_cp: u32) -> Option<i8> {
    if self.kerning_off == 0 || self.kerning_count == 0 {
      return None;
    }
    let mut off = self.kerning_off;
    for _ in 0..self.kerning_count {
      if off + 7 > self.data.len() {
        break;
      }
      let l = le_u24(&self.data[off..off + 3]);
      let r = le_u24(&self.data[off + 3..off + 6]);
      let adj = self.data[off + 6] as i8;
      off += 7;
      if l == left_cp && r == right_cp {
        return Some(adj);
      }
    }
    None
  }

  /// Create a streaming glyph iterator for a glyph index.
  pub fn glyph_iter_for_index(&'a self, index: u16) -> Option<GlyphIter<'a>> {
    let meta = self.glyph_meta(index)?;
    Some(GlyphIter::new(self, meta))
  }

  /// Create a streaming glyph iterator for a Unicode codepoint.
  pub fn glyph_iter_for_cp(&'a self, cp: u32) -> Option<GlyphIter<'a>> {
    let idx = self.glyph_index_for_cp(cp)?;
    self.glyph_iter_for_index(idx)
  }

  #[inline]
  fn row_present(&self, y: u16) -> bool {
    let bit = y as usize;
    if bit >= self.atlas_height as usize {
      return false;
    }
    let byte = self.data[self.row_mask_off + (bit / 8)];
    let mask = 1u8 << (bit % 8);
    (byte & mask) != 0
  }
}

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
  fn new(font: &'a MiniTypeFont<'a>, meta: GlyphMeta) -> Self {
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
        return Some(self.font.palette[0]);
      }
    }
  }
}

// ---------- helpers ----------

#[inline]
fn le_u16(b: &[u8]) -> u16 {
  u16::from_le_bytes(b[0..2].try_into().unwrap())
}

#[inline]
fn le_i16(b: &[u8]) -> i16 {
  i16::from_le_bytes(b[0..2].try_into().unwrap())
}

#[inline]
fn le_u24(b: &[u8]) -> u32 {
  (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16)
}

#[inline]
fn le_u32(b: &[u8]) -> u32 {
  u32::from_le_bytes(b[0..4].try_into().unwrap())
}

#[inline]
fn ceil_div_u16(v: u16, d: u16) -> u16 {
  (v + (d - 1)) / d
}

fn count_present_rows(mask: &[u8], height: u16) -> u16 {
  // Count set bits up to `height`.
  let mut cnt = 0u16;
  let mut y = 0u16;
  let mut i = 0usize;
  while y < height {
    let byte = mask[i];
    let bits = core::cmp::min(8, (height - y) as usize) as u8;
    let mut b = byte;
    for _ in 0..bits {
      cnt += (b & 1) as u16;
      b >>= 1;
    }
    y += bits as u16;
    i += 1;
  }
  cnt
}
