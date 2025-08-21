#![no_std]

mod glyph;
mod style;
pub use glyph::*;
pub use style::MiniTextStyle;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Error {
  BadMagic,
  BadVersion(u8),
  Malformed,
  KerningBounds,
  AtlasTooShort,
  GlyphOob,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MiniTypeFont<'a> {
  pub line_height: u16,
  pub ascent: i16,
  pub descent: i16,

  data: &'a [u8],
  charsets_count: u16,
  glyph_count: u16,
  kerning_count: u32,

  glyph_table_off: usize,
  atlas_off: usize,
  kerning_off: usize,
}

const VERSION: u8 = 0;

impl<'a> MiniTypeFont<'a> {
  pub fn new(data: &'a [u8]) -> Result<Self, Error> {
    // Minimal prefix: magic(4) + ver(1) + flags(1) + lh(1) + asc(1) + desc(1) + seg_cnt(1) = 10 bytes
    if data.len() < 10 {
      return Err(Error::Malformed);
    }
    if &data[0..4] != b"MFNT" {
      return Err(Error::BadMagic);
    }
    let version = data[4];
    if data[4] != VERSION {
      return Err(Error::BadVersion(version));
    }
    if data[5] != 0 {
      return Err(Error::Malformed); // reserved for future use
    }

    let line_height = data[6] as u16;
    let ascent = (data[7] as i8) as i16;
    let descent = (data[8] as i8) as i16;
    let charsets_count = data[9] as u16;

    // Segments: start at 10, each 5 bytes { u24 start_cp, u16 len }
    let segs_len = (charsets_count as usize).checked_mul(5).ok_or(Error::Malformed)?;
    let segs_end = 10usize.checked_add(segs_len).ok_or(Error::Malformed)?;
    if segs_end + 2 > data.len() {
      return Err(Error::Malformed);
    }

    // glyph_count follows segments
    let glyph_count = le_u16_at(data, segs_end);
    let glyph_table_off = segs_end + 2;
    let glyph_table_len = (glyph_count as usize).checked_mul(6).ok_or(Error::Malformed)?;
    let gt_end = glyph_table_off.checked_add(glyph_table_len).ok_or(Error::Malformed)?;
    if gt_end > data.len() {
      return Err(Error::Malformed);
    }

    // Atlas blob starts after glyph table
    let atlas_off = glyph_table_off + glyph_table_len;

    // Derive atlas_len by scanning glyphs (max end)
    let atlas_end = {
      let mut max_end = atlas_off;
      let mut i = 0usize;
      while i < glyph_table_len {
        let meta = GlyphMetrics::decode(&data[glyph_table_off..], i).ok_or(Error::GlyphOob)?;
        let bpr = ceil_div_u16(meta.w as u16, 2) as usize; // L4 (2px/byte)
        let need = (meta.h as usize).saturating_mul(bpr);
        let start = atlas_off + meta.offset as usize;
        let end = start.checked_add(need).ok_or(Error::AtlasTooShort)?;
        if end > max_end {
          max_end = end;
        }
        i += 6;
      }
      max_end
    };
    if atlas_end > data.len() {
      return Err(Error::AtlasTooShort);
    }

    // Optional kerning tail: [u16 count][count × (u16,u16,i8)]
    let (kerning_off, kerning_count) = if atlas_end < data.len() {
      if atlas_end + 2 > data.len() {
        return Err(Error::KerningBounds);
      }
      let kc = le_u16_at(data, atlas_end) as usize;
      let need_pairs = kc.checked_mul(5).ok_or(Error::KerningBounds)?;
      let total_need = 2usize.checked_add(need_pairs).ok_or(Error::KerningBounds)?;
      if atlas_end + total_need != data.len() {
        return Err(Error::KerningBounds);
      }
      (atlas_end + 2, kc as u32)
    } else {
      (data.len(), 0)
    };

    Ok(Self {
      data,
      charsets_count,
      glyph_count,
      line_height,
      ascent,
      descent,
      glyph_table_off,
      atlas_off,
      kerning_off,
      kerning_count,
    })
  }

  /// Lightweight constructor (no validation). Offsets/lengths that require scanning (atlas_len, kerning) are zeroed.
  #[inline]
  pub const fn unchecked(data: &'a [u8]) -> Self {
    let charsets_count = data[9] as u16;
    let charsets_end = 10 + (charsets_count as usize) * 5;

    let glyph_count = le_u16_at(data, charsets_end);
    let glyph_table_off = charsets_end + 2;
    let glyph_table_len = (glyph_count as usize) * 6;
    let atlas_off = glyph_table_off + glyph_table_len;

    Self {
      data,
      charsets_count,
      glyph_count,
      line_height: data[6] as u16,
      ascent: (data[7] as i8) as i16,
      descent: (data[8] as i8) as i16,
      glyph_table_off,
      atlas_off,
      kerning_off: data.len(), // none
      kerning_count: 0,
    }
  }

  /// Map a `char` to a `GlyphId` using compact segments.
  #[inline(always)]
  pub fn glyph(&self, char: char) -> Option<GlyphId> {
    let cp = char as u32;
    let mut offset = 10usize; // start of segments
    let mut base: u32 = 0;
    for _ in 0..self.charsets_count {
      let start = le_u24_at(self.data, offset);
      let len = le_u16_at(self.data, offset + 3) as u32;
      offset += 5;

      if cp >= start && cp < start + len {
        return Some(GlyphId((base + (cp - start)) as u16));
      }

      base += len;
    }
    None
  }

  #[inline(always)]
  pub fn metrics(&self, id: GlyphId) -> Option<GlyphMetrics> {
    let idx = id.0 as usize;
    if idx >= self.glyph_count as usize {
      return None;
    }
    let off = self.glyph_table_off + idx * 6;
    GlyphMetrics::decode(self.data, off)
  }

  /// Kerning between two glyph ids (0 if absent).
  #[inline(always)]
  pub fn kerning(&self, left: GlyphId, right: GlyphId) -> i8 {
    let kc = self.kerning_count;
    if kc == 0 {
      return 0;
    }
    let mut off = self.kerning_off;
    // Linear scan — table is typically small. Keep tight & inlinable.
    for _ in 0..kc {
      // Bounds guaranteed by validated constructor.
      let l = le_u16_at(self.data, off);
      let r = le_u16_at(self.data, off + 2);
      let adj = self.data[off + 4] as i8;
      if l == left.0 && r == right.0 {
        return adj;
      }
      off += 5;
    }
    0
  }

  /// Fast, concrete iterator over all L4 pixels (0..=15) of a glyph, row-major.
  /// Returns `None` if the glyph is missing. Yields exactly `w*h` items.
  ///
  /// This is the fastest safe path if you need per-pixel values (e.g., LUT -> alpha).
  #[inline(always)]
  pub(crate) fn glyph_pixels(&'a self, id: GlyphId) -> Option<GlyphPixels<'a>> {
    let m = self.metrics(id)?;
    let bpr = ceil_div_u16(m.w as u16, 2) as usize;
    // Start of the glyph's slab in the atlas (contiguous rows)
    let start = self.atlas_off + m.offset as usize;
    let len = (m.h as usize).saturating_mul(bpr);
    // Slice of exactly this glyph's rows (L4; 2px per byte)
    let slab = &self.data[start..start + len];
    Some(GlyphPixels::new(slab, m))
  }
}

// ---------- decode & helpers ----------

#[inline]
pub const fn ceil_div_u16(v: u16, d: u16) -> u16 {
  (v + (d - 1)) / d
}
#[inline]
pub const fn le_u16_at(b: &[u8], off: usize) -> u16 {
  (b[off] as u16) | ((b[off + 1] as u16) << 8)
}
#[inline]
pub const fn le_u24_at(b: &[u8], off: usize) -> u32 {
  (b[off] as u32) | ((b[off + 1] as u32) << 8) | ((b[off + 2] as u32) << 16)
}
