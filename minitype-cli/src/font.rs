//! MiniType (v0, compact header; reordered tables)
//!
//! Order:
//!   "MFNT"(4), u8 version=0, u8 flags=0 (reserved)
//!   u8 line_height, i8 ascent, i8 descent,
//!   u8 segment_count,
//!   segment_count × { u24 start_cp, u16 len },                // 5 bytes/segment
//!   u16 glyph_count,
//!   glyph_table   (glyph_count × 6 bytes/rec),
//!   atlas_blob,
//!   [ u16 kerning_count,                                    // OPTIONAL
//!     kerning_table (kerning_count × 5 bytes/pair) ]        // {u16,u16,i8}
//!
//! Glyph record (6 bytes, little-endian bit packing):
//!   u7 w, u7 h, u7 yoff, i5 adv_delta_to_w, i4 left_bearing, u18 blob_off
//!
//! L4: 2 px/byte, low nibble = left pixel (0..15).

use core::convert::TryInto;

const MAGIC: &[u8; 4] = b"MFNT";
const VERSION: u8 = 0;
const L4_PIXELS_PER_BYTE: usize = 2;

// ──────────────────────────────────────────────────────────────────────────────
// metadata
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Metadata {
  pub line_height: u16,
  pub ascent: i16,
  pub descent: i16,
  pub glyphs: Vec<Glyph>,
  #[serde(default)]
  pub kerning: Vec<Kerning>,
  #[serde(default)]
  pub charset: Vec<CharsetRange>,
}

impl Metadata {
  #[inline]
  pub fn line_height_u8(&self) -> anyhow::Result<u8> {
    if (1..=255).contains(&self.line_height) {
      Ok(self.line_height as u8)
    } else {
      anyhow::bail!("line_height={} out of u8 (1..=255)", self.line_height)
    }
  }
  #[inline]
  pub fn ascent_i8(&self) -> anyhow::Result<i8> {
    if (-128..=127).contains(&self.ascent) {
      Ok(self.ascent as i8)
    } else {
      anyhow::bail!("ascent={} out of i8", self.ascent)
    }
  }
  #[inline]
  pub fn descent_i8(&self) -> anyhow::Result<i8> {
    if (-128..=127).contains(&self.descent) {
      Ok(self.descent as i8)
    } else {
      anyhow::bail!("descent={} out of i8", self.descent)
    }
  }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Glyph {
  pub x: u16, // column in source A8 atlas
  pub w: u8,  // tight width (px)
  #[serde(default)]
  pub advance: Option<i8>,
  #[serde(default)]
  pub left: Option<i8>,
}
impl Glyph {
  pub fn new(x: u16, w: u8, advance: Option<i8>, left: Option<i8>) -> Self {
    Self { x, w, advance, left }
  }
  #[inline]
  fn advance_or_width(&self) -> i32 {
    self.advance.map(|v| v as i32).unwrap_or(self.w as i32)
  }
  #[inline]
  fn left_bearing_or_zero(&self) -> i32 {
    self.left.map(|v| v as i32).unwrap_or(0)
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// kerning & charset
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Kerning {
  pub left: String,
  pub right: String,
  pub adj: i8,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CharsetRange {
  pub start: String,
  pub end: String,
}

impl CharsetRange {
  #[inline]
  fn span(&self) -> anyhow::Result<(u32, u32, u16)> {
    let s = one_scalar(&self.start, "charset.start")?;
    let e = one_scalar(&self.end, "charset.end")?;
    if e < s {
      anyhow::bail!("range end {:#X} < start {:#X}", e, s);
    }
    let len_u32 = e - s + 1;
    Ok((s, e, u16::try_from(len_u32).map_err(|_| anyhow::anyhow!("range len {} > u16::MAX", len_u32))?))
  }
}

struct KerningTable {
  bytes: Vec<u8>,
  count: u32,
}
impl KerningTable {
  /// Always encodes { u16 left_idx, u16 right_idx, i8 adj } (5 bytes).
  fn from_pairs(kerning: &[Kerning], segs: &CharsetSegments, glyph_count: usize) -> anyhow::Result<Self> {
    if kerning.is_empty() {
      return Ok(Self { bytes: Vec::new(), count: 0 });
    }
    if glyph_count > u16::MAX as usize {
      anyhow::bail!("glyph_count {} exceeds u16 addressable range", glyph_count);
    }
    let mut bytes = Vec::with_capacity(kerning.len() * 5);
    for (i, k) in kerning.iter().enumerate() {
      let lcp = one_scalar(&k.left, &format!("kerning[{i}].left"))?;
      let rcp = one_scalar(&k.right, &format!("kerning[{i}].right"))?;
      let li = segs
        .glyph_index_of(lcp)
        .ok_or_else(|| anyhow::anyhow!("kerning[{i}] left U+{lcp:04X} not in charset"))?;
      let ri = segs
        .glyph_index_of(rcp)
        .ok_or_else(|| anyhow::anyhow!("kerning[{i}] right U+{rcp:04X} not in charset"))?;
      bytes.extend_from_slice(&li.to_le_bytes());
      bytes.extend_from_slice(&ri.to_le_bytes());
      bytes.push(k.adj as u8);
    }
    Ok(Self { bytes, count: kerning.len() as u32 })
  }
}

struct CharsetSegments {
  items: Vec<(u32, u16)>, // (start_cp, len) ; base is implicit
  count: u16,
}
impl CharsetSegments {
  fn build(charset: &[CharsetRange], glyph_len: usize) -> anyhow::Result<Self> {
    if charset.is_empty() {
      anyhow::bail!("charset is required");
    }
    let mut items = Vec::with_capacity(charset.len());
    let mut total_len: u32 = 0;
    for (i, r) in charset.iter().enumerate() {
      let (start, _end, len) = r.span().map_err(|e| anyhow::anyhow!("charset[{i}]: {e}"))?;
      total_len = total_len
        .checked_add(len as u32)
        .ok_or_else(|| anyhow::anyhow!("charset length overflow (+{})", len))?;
      items.push((start, len));
    }
    if total_len as usize != glyph_len {
      anyhow::bail!("charset codepoints {} != glyphs {}", total_len, glyph_len);
    }
    let count: u16 = items
      .len()
      .try_into()
      .map_err(|_| anyhow::anyhow!("too many charset segments"))?;
    Ok(Self { items, count })
  }

  #[inline]
  fn glyph_index_of(&self, cp: u32) -> Option<u16> {
    let mut base: u32 = 0;
    for (start, len) in &self.items {
      let len_u32 = *len as u32;
      if cp >= *start && cp < *start + len_u32 {
        return u16::try_from(base + (cp - *start)).ok();
      }
      base += len_u32;
    }
    None
  }

  fn write_to(&self, out: &mut Vec<u8>) {
    for (start_cp, len) in &self.items {
      push_u24(out, *start_cp);
      out.extend_from_slice(&len.to_le_bytes());
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
/* Glyph record packing */
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct GlyphMetrics {
  w: u8,
  h: u8,
  yoff: u8,
  adv_i5: i8,
  left_i4: i8,
  off18: u32,
}
impl GlyphMetrics {
  #[inline]
  fn clamp_i5(v: i32) -> i8 {
    if v < -16 {
      -16
    } else if v > 15 {
      15
    } else {
      v as i8
    }
  }
  #[inline]
  fn clamp_i4(v: i32) -> i8 {
    if v < -8 {
      -8
    } else if v > 7 {
      7
    } else {
      v as i8
    }
  }

  fn try_from_glyph(g: &Glyph, h_u7: u8, yoff_u7: u8, off_bytes: u32) -> anyhow::Result<Self> {
    if g.w > 127 {
      anyhow::bail!("width {} exceeds u7", g.w);
    }
    if h_u7 > 127 {
      anyhow::bail!("height {} exceeds u7", h_u7);
    }
    if yoff_u7 > 127 {
      anyhow::bail!("yoff {} exceeds u7", yoff_u7);
    }
    if off_bytes > 0x3FFFF {
      anyhow::bail!("blob off {} exceeds u18", off_bytes);
    }
    let adv_delta = g.advance_or_width() - (g.w as i32);
    Ok(Self {
      w: g.w,
      h: h_u7,
      yoff: yoff_u7,
      adv_i5: Self::clamp_i5(adv_delta),
      left_i4: Self::clamp_i4(g.left_bearing_or_zero()),
      off18: off_bytes,
    })
  }

  #[inline]
  fn to_bytes(self) -> [u8; 6] {
    debug_assert!(self.w <= 127 && self.h <= 127 && self.yoff <= 127 && self.off18 <= 0x3FFFF);
    let adv5 = ((self.adv_i5 as i16) & 0x1F) as u64;
    let left4 = ((self.left_i4 as i16) & 0x0F) as u64;
    let val: u64 = (self.w as u64)
      | ((self.h as u64) << 7)
      | ((self.yoff as u64) << 14)
      | (adv5 << 21)
      | (left4 << 26)
      | (((self.off18 as u64) & 0x3FFFF) << 30);
    [
      (val & 0xFF) as u8,
      ((val >> 8) & 0xFF) as u8,
      ((val >> 16) & 0xFF) as u8,
      ((val >> 24) & 0xFF) as u8,
      ((val >> 32) & 0xFF) as u8,
      ((val >> 40) & 0xFF) as u8,
    ]
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// atlas & builder
// ──────────────────────────────────────────────────────────────────────────────

pub struct AtlasBlob {
  pub glyph_table: Vec<u8>,
  pub atlas_blob: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Atlas {
  pub width: u16,
  pub height: u16,
  pub pixels: Vec<u8>,
} // A8, row-major

impl Atlas {
  #[inline]
  pub fn new(width: u16, height: u16, pixels: Vec<u8>) -> anyhow::Result<Self> {
    let need = width as usize * height as usize;
    if pixels.len() != need {
      anyhow::bail!("atlas len {} != {}*{}", pixels.len(), width, height);
    }
    Ok(Self { width, height, pixels })
  }

  fn tight_bounds_for_span(&self, x0: usize, gw: usize) -> anyhow::Result<(u8, u8, usize, usize)> {
    let w_atlas = self.width as usize;
    let h_atlas = self.height as usize;
    if x0 + gw > w_atlas {
      anyhow::bail!("span out of bounds (x0={}, gw={}, w={})", x0, gw, w_atlas);
    }

    let mut y_min = h_atlas;
    let mut y_max = 0usize;
    for y in 0..h_atlas {
      let row = &self.pixels[y * w_atlas + x0..y * w_atlas + x0 + gw];
      if row.iter().any(|&px| px != 0) {
        if y < y_min {
          y_min = y;
        }
        if y > y_max {
          y_max = y;
        }
      }
    }
    if y_min > y_max {
      return Ok((0, 0, 0, 0));
    } // empty

    let h = (y_max - y_min + 1) as i32;
    if h > 127 {
      anyhow::bail!("tight height {} exceeds u7", h);
    }
    let yoff = y_min as i32;
    if yoff > 127 {
      anyhow::bail!("y_off {} exceeds u7", yoff);
    }
    Ok((h as u8, yoff as u8, y_min, y_max))
  }

  fn encode_rows_l4_into(&self, x0: usize, gw: usize, y_min: usize, y_max: usize, blob: &mut Vec<u8>) {
    #[inline]
    fn quant_l4_round(v: u8) -> u8 {
      let idx = ((v as u16 + 8) / 16) as u16;
      if idx > 15 { 15 } else { idx as u8 }
    }
    let w_atlas = self.width as usize;
    let bytes_per_row = (gw + 1) / L4_PIXELS_PER_BYTE;
    blob.reserve((y_max.saturating_sub(y_min) + 1) * bytes_per_row);
    for y in y_min..=y_max {
      let base = y * w_atlas + x0;
      let mut x = 0usize;
      while x < gw {
        let il = quant_l4_round(self.pixels[base + x]) & 0x0F;
        let ir = if x + 1 < gw {
          quant_l4_round(self.pixels[base + x + 1]) & 0x0F
        } else {
          0
        };
        blob.push((ir << 4) | il);
        x += 2;
      }
    }
  }

  pub fn build(&self, meta: &Metadata) -> anyhow::Result<AtlasBlob> {
    if meta.glyphs.is_empty() {
      anyhow::bail!("glyphs are required");
    }
    let mut glyph_table = Vec::with_capacity(meta.glyphs.len() * 6);
    let mut atlas_blob = Vec::<u8>::new();

    for (i, g) in meta.glyphs.iter().enumerate() {
      let x0 = g.x as usize;
      let gw = g.w as usize;
      let (h_u7, yoff_u7, y_min, y_max) = self.tight_bounds_for_span(x0, gw)?;
      let off_bytes = atlas_blob.len() as u32;
      if off_bytes > 0x3FFFF {
        anyhow::bail!("glyph[{i}] blob off {} exceeds u18", off_bytes);
      }
      let rec = GlyphMetrics::try_from_glyph(g, h_u7, yoff_u7, off_bytes)?;
      glyph_table.extend_from_slice(&rec.to_bytes());
      if h_u7 > 0 {
        self.encode_rows_l4_into(x0, gw, y_min, y_max, &mut atlas_blob);
      }
    }
    Ok(AtlasBlob { glyph_table, atlas_blob })
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// assemble (compact header writer) — fixed strides, version=0, flags=0
// ──────────────────────────────────────────────────────────────────────────────

pub fn assemble(meta: Metadata, atlas: Atlas) -> anyhow::Result<Vec<u8>> {
  let (lh_u8, asc_i8, desc_i8) = (meta.line_height_u8()?, meta.ascent_i8()?, meta.descent_i8()?);

  let AtlasBlob { glyph_table, atlas_blob } = atlas.build(&meta)?;
  let glyph_count_u16: u16 = meta
    .glyphs
    .len()
    .try_into()
    .map_err(|_| anyhow::anyhow!("too many glyphs"))?;

  let segs = CharsetSegments::build(&meta.charset, meta.glyphs.len())?;
  let kern = KerningTable::from_pairs(&meta.kerning, &segs, meta.glyphs.len())?;

  let seg_cnt_u8: u8 = segs
    .count
    .try_into()
    .map_err(|_| anyhow::anyhow!("segment_count > 255"))?;
  let ker_cnt_u16: u16 = kern
    .count
    .try_into()
    .map_err(|_| anyhow::anyhow!("kerning_count > 65535"))?;

  // prefix = 10 bytes: magic(4) + ver(1) + flags(1=0) + lh(1) + asc(1) + desc(1) + seg_cnt(1)
  let header_prefix_len = 10usize;
  let segments_len = (segs.count as usize) * 5; // 5 bytes/segment
  let glyph_count_field_len = 2usize;
  let kerning_prefix_len = if kern.count > 0 { 2 } else { 0 }; // u16 kerning_count if present

  let mut out = Vec::with_capacity(
    header_prefix_len
      + segments_len
      + glyph_count_field_len
      + glyph_table.len()
      + atlas_blob.len()
      + kerning_prefix_len
      + kern.bytes.len(),
  );

  out.extend_from_slice(MAGIC);
  out.push(VERSION);
  out.push(0); // flags (reserved)
  out.push(lh_u8);
  out.push(asc_i8 as u8);
  out.push(desc_i8 as u8);

  // -- segment_count and segments
  out.push(seg_cnt_u8);
  segs.write_to(&mut out);

  // -- glyph_count then glyph table
  out.extend_from_slice(&glyph_count_u16.to_le_bytes());
  out.extend_from_slice(&glyph_table);

  // -- atlas blob
  out.extend_from_slice(&atlas_blob);

  // -- optional kerning tail (5B/pair)
  if kern.count > 0 {
    out.extend_from_slice(&ker_cnt_u16.to_le_bytes());
    out.extend_from_slice(&kern.bytes);
  }

  Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// small helpers
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn one_scalar(s: &str, label: &str) -> anyhow::Result<u32> {
  let mut it = s.chars();
  let c = it.next().ok_or_else(|| anyhow::anyhow!("{label} is empty"))?;
  if it.next().is_some() {
    anyhow::bail!("{label} must be one scalar");
  }
  Ok(c as u32)
}
#[inline]
fn push_u24(buf: &mut Vec<u8>, v: u32) {
  buf.extend_from_slice(&[(v & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, ((v >> 16) & 0xFF) as u8]);
}
