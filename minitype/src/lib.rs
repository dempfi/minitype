#![no_std]

//! Minimal no_std MiniType font reader & streaming glyph iterator (multi-palette L4).
//!
//! File format (relevant parts):
//! - Header "MFNT", version=1
//! - Glyph table: 6 bytes/record → u16 x, u8 w, i8 advance, i8 left, u8 q
//! - Atlas blob: u16 w, u16 h, [u8;64] palettes (4×16; each palette[0]==0),
//!               rowmask[ceil(h/8)], rows (L4, 2px/byte)
//!
//! This module is `no_std` (uses only `core`).

mod glyph;
mod style;
use glyph::GlyphIter;

pub use style::MiniTextStyle;

/// Parsing/validation errors
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum Error {
  /// Wrong magic (expected "MFNT").
  BadMagic,
  /// Unsupported version (expected 1).
  BadVersion(u8),
  /// Header/glyph-table alignment/length errors.
  Malformed,
  /// Declared total length does not match input length.
  LengthMismatch,
  /// Inner atlas header too short or inconsistent.
  AtlasHeader,
  /// Any atlas palette[k][0] must be 0x00.
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

/// Compact glyph metadata (6 bytes/glyph in the file).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlyphMeta {
  pub x: u16,
  pub w: u8,
  pub advance: i8,
  pub left: i8,
  /// Quantization/dequantization hints byte:
  pub palette_id: usize,
  pub scale_idx: usize,
  pub bias16: u16,
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
  pub glyph_count: u16,
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
  palettes: [[u8; 16]; 4], // 4 palettes
  row_mask_off: usize,
  row_mask_len: usize,     // == ceil(h/8)
  payload_off: usize,      // == atlas_off + 68 + row_mask_len
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
    let line_height = le_u16_at(data, 6);
    let ascent = le_i16_at(data, 8);
    let descent = le_i16_at(data, 10);
    let glyph_count = le_u16_at(data, 12);
    let glyph_table_off = le_u32_at(data, 14) as usize;
    let glyph_table_len = le_u32_at(data, 18) as usize;
    let atlas_off = le_u32_at(data, 22) as usize;
    let atlas_len = le_u32_at(data, 26) as usize;
    let total_len = le_u32_at(data, 30) as usize;
    let kerning_off = le_u32_at(data, 34) as usize;
    let kerning_count = le_u32_at(data, 38);
    let charset_seg_count = le_u16_at(data, 42);

    if total_len != data.len() {
      return Err(Error::LengthMismatch);
    }

    let charset_segs_off = 0x2C;
    let header_len = 0x2C + 7usize * (charset_seg_count as usize);
    if glyph_table_off != header_len {
      return Err(Error::Malformed);
    }
    // 6 bytes per glyph now
    if glyph_table_len != 6usize * (glyph_count as usize) {
      return Err(Error::Malformed);
    }
    if atlas_off != glyph_table_off + glyph_table_len {
      return Err(Error::Malformed);
    }
    // Atlas inner header must have at least 68 bytes (w,h,4×16 palettes)
    if atlas_off + 68 > total_len {
      return Err(Error::AtlasHeader);
    }

    // ---- Inner atlas header ----
    let width = le_u16_at(data, atlas_off);
    let height = le_u16_at(data, atlas_off + 2);

    // Read 4×16 palettes
    let mut palettes = [[0u8; 16]; 4];
    let mut p_off = atlas_off + 4;
    for k in 0..4 {
      palettes[k] = read_u8_16(data, p_off);
      p_off += 16;
      if palettes[k][0] != 0 {
        return Err(Error::Palette0);
      }
    }

    let row_mask_len = ceil_div_u16(height, 8) as usize;
    let row_mask_off = atlas_off + 68;
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
    if atlas_len < 68 + row_mask_len + need_payload {
      return Err(Error::AtlasTooShort);
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
      palettes,
      row_mask_off,
      row_mask_len,
      payload_off,
      row_stride_bytes,
    })
  }

  /// Construct a `MiniTypeFont` view **without validation**.
  ///
  /// # Safety
  /// Caller must guarantee that `data` points to a well-formed MiniType
  /// font blob that adheres to the spec.
  pub const fn raw(data: &'a [u8]) -> Self {
    let line_height = le_u16_at(data, 6);
    let ascent = le_i16_at(data, 8);
    let descent = le_i16_at(data, 10);
    let glyph_count = le_u16_at(data, 12);
    let glyph_table_off = le_u32_at(data, 14) as usize;
    let glyph_table_len = le_u32_at(data, 18) as usize;
    let atlas_off = le_u32_at(data, 22) as usize;
    let atlas_len = le_u32_at(data, 26) as usize;
    let _total_len = le_u32_at(data, 30) as usize;
    let kerning_off = le_u32_at(data, 34) as usize;
    let kerning_count = le_u32_at(data, 38);
    let charset_seg_count = le_u16_at(data, 42);

    // Derived header layout values
    let charset_segs_off = 0x2C;
    let header_len = 0x2C + 7usize * (charset_seg_count as usize);

    // Inner atlas header (no validation)
    let width = le_u16_at(data, atlas_off);
    let height = le_u16_at(data, atlas_off + 2);

    // 4×16 palettes
    let mut palettes = [[0u8; 16]; 4];
    let mut p_off = atlas_off + 4;
    let mut k = 0;
    while k < 4 {
      palettes[k] = read_u8_16(data, p_off);
      p_off += 16;
      k += 1;
    }

    // Row mask / payload geometry (no validation)
    let row_mask_off = atlas_off + 68;
    let row_mask_len = ceil_div_u16(height, 8) as usize;
    let payload_off = row_mask_off + row_mask_len;
    let row_stride_bytes = ceil_div_u16(width, 2) as usize;

    Self {
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
      palettes,
      row_mask_off,
      row_mask_len,
      payload_off,
      row_stride_bytes,
    }
  }

  /// Fetch glyph metadata by glyph index (6 bytes/record).
  #[inline]
  pub fn glyph_meta(&self, index: u16) -> Option<GlyphMeta> {
    if index as usize >= self.glyph_count as usize {
      return None;
    }
    let off = self.glyph_table_off + (index as usize) * 6;
    if off + 6 > self.data.len() {
      return None;
    }
    let x = le_u16_at(&self.data, off);
    let w = self.data[off + 2];
    let advance = self.data[off + 3] as i8;
    let left = self.data[off + 4] as i8;
    let q = self.data[off + 5];
    let palette_id = (q & 0b11) as usize;
    let scale_idx = ((q >> 2) & 0b11) as usize;
    let bias16 = ((q >> 4) & 0x0F) as u16;
    Some(GlyphMeta { x, w, advance, left, palette_id, scale_idx, bias16 })
  }

  /// Map Unicode scalar value to glyph index using charset segments.
  /// Returns `None` if the codepoint is not covered.
  pub fn glyph_index_for_cp(&self, cp: u32) -> Option<u16> {
    let mut off = self.charset_segs_off;
    for _ in 0..self.charset_seg_count {
      if off + 7 > self.data.len() {
        return None;
      }
      let start = le_u24_at(&self.data, off);
      let len = le_u16_at(&self.data, off + 3);
      let base = le_u16_at(&self.data, off + 5);
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
      let l = le_u24_at(&self.data, off);
      let r = le_u24_at(&self.data, off + 3);
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
  pub fn palettes(&self) -> &[[u8; 16]; 4] {
    &self.palettes
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

#[inline]
pub const fn ceil_div_u16(v: u16, d: u16) -> u16 {
  (v + (d - 1)) / d
}

#[inline]
pub const fn le_u16_at(b: &[u8], off: usize) -> u16 {
  (b[off] as u16) | ((b[off + 1] as u16) << 8)
}

#[inline]
pub const fn le_i16_at(b: &[u8], off: usize) -> i16 {
  i16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
pub const fn le_u24_at(b: &[u8], off: usize) -> u32 {
  (b[off] as u32) | ((b[off + 1] as u32) << 8) | ((b[off + 2] as u32) << 16)
}

#[inline]
pub const fn le_u32_at(b: &[u8], off: usize) -> u32 {
  (b[off] as u32) | ((b[off + 1] as u32) << 8) | ((b[off + 2] as u32) << 16) | ((b[off + 3] as u32) << 24)
}

#[inline]
pub const fn read_u8_16(b: &[u8], off: usize) -> [u8; 16] {
  let mut out = [0u8; 16];
  let mut i = 0;
  while i < 16 {
    out[i] = b[off + i];
    i += 1;
  }
  out
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
