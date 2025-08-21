//! MiniType font: compact, MCU-friendly container + per-glyph L4-bytes RLE atlas.
//!
//! Atlas blob layout (Per-Glyph RLE over **bytes**, each byte = 2 L4 pixels):
//!   u16 w_total, u16 h_tight, u16 y_off,
//!   glyph_0_rle, glyph_1_rle, … glyph_{N-1}_rle
//!
//!   For glyph i of width w_i, the encoder packs rows as L4 bytes:
//!     byte = (right<<4) | left, low nibble = left pixel
//!   For odd w_i, the last byte of each row has high nibble = 0.
//!   The RLE stream is over these BYTES (row-major, contiguous rows).
//!
//! RLE packets (BYTES):
//!   - Literal: opcode in [0x00..0x7F] => len = opcode+1 bytes; then `len` raw bytes.
//!   - Run:     opcode in [0x80..0xFF] => len = (op&0x7F)+1 bytes; then 1 byte value to repeat.
//!     (Encoder uses run threshold >= 3 bytes; max len per packet = 128.)
//!
//! MFNT layout:
//!   "MFNT", u8 version=1, u8 flags,            // flags bit1=1 -> PerGlyph RLE L4-bytes
//!   u16 line_height, i16 ascent, i16 descent,
//!   u16 glyph_count,
//!   u32 glyphs_off, u32 glyphs_len,
//!   u32 atlas_off,  u32 atlas_len,
//!   u32 total_len (backpatch),
//!   u32 kerning_off (backpatch), u32 kerning_cnt (backpatch),
//!   u16 segment_count,
//!   segment_count × { u24 start_cp, u16 len, u16 base },
//!   glyph table (u16 x, u8 w, i8 advance, i8 left, u16 next_off), // 7 B/record
//!   atlas blob,
//!   kerning table (u24 left_cp, u24 right_cp, i8 adj).

use core::convert::TryInto;

const MAGIC: &[u8; 4] = b"MFNT";
const VERSION: u8 = 1;

/// MFNT header bytes before charset segments start (fixed-size prefix).
const HDR_PREFIX_LEN: usize = 0x2C;

/// MFNT.flags bit1: atlas is Per-Glyph RLE L4 **bytes** (bit0 reserved).
const MFNT_FLAG_ATLAS_PGRLE: u8 = 0b0000_0010;

#[inline]
fn quant_l4_round(v: u8) -> u8 {
  let idx = ((v as u16 + 8) / 16) as u16;
  if idx > 15 { 15 } else { idx as u8 }
}

// ──────────────────────────────────────────────────────────────────────────────
// metadata types
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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Glyph {
  pub x: u16,
  pub w: u8,
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
  fn advance_or_width(&self) -> i8 {
    self.advance.unwrap_or(self.w as i8)
  }
  #[inline]
  fn left_bearing_or_zero(&self) -> i8 {
    self.left.unwrap_or(0)
  }
}

#[inline]
fn glyph_to_bytes_with_next(g: &Glyph, next_off: u16) -> [u8; 7] {
  let mut out = [0u8; 7];
  out[..2].copy_from_slice(&g.x.to_le_bytes());
  out[2] = g.w;
  out[3] = g.advance_or_width() as u8; // i8 two's complement
  out[4] = g.left_bearing_or_zero() as u8; // i8 two's complement
  out[5..7].copy_from_slice(&next_off.to_le_bytes());
  out
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Kerning {
  pub left: String,
  pub right: String,
  pub adj: i8,
}
impl Kerning {
  #[inline]
  pub fn to_bytes(&self) -> anyhow::Result<[u8; 7]> {
    let l = one_scalar(&self.left, "kerning.left")?;
    let r = one_scalar(&self.right, "kerning.right")?;
    let mut out = [0u8; 7];
    put_u24(&mut out[0..3], l);
    put_u24(&mut out[3..6], r);
    out[6] = self.adj as u8;
    Ok(out)
  }
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
  #[inline]
  fn segment_with_base(&self, base: u16) -> anyhow::Result<((u32, u16, u16), u16)> {
    let (start, _end, len) = self.span()?;
    let next = base
      .checked_add(len)
      .ok_or_else(|| anyhow::anyhow!("glyph base overflow (+{})", len))?;
    Ok(((start, len, base), next))
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// atlas source (A8 pixels)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Atlas {
  pub width: u16,
  pub height: u16,
  pub pixels: Vec<u8>, // A8 source (linear)
}
impl Atlas {
  #[inline]
  pub fn new(width: u16, height: u16, pixels: Vec<u8>) -> anyhow::Result<Self> {
    let need = width as usize * height as usize;
    if pixels.len() != need {
      anyhow::bail!("atlas len {} != {}*{}", pixels.len(), width, height);
    }
    Ok(Self { width, height, pixels })
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-glyph RLE over L4 BYTES (2 px/byte)
// ──────────────────────────────────────────────────────────────────────────────

fn build_pgrle_blob(a: &Atlas, glyphs: &[Glyph]) -> (Vec<u8>, Vec<u16>, u16, u16, u16) {
  let w_total = a.width as usize;
  let h_total = a.height as usize;

  // Tight vertical band (shared).
  let mut y_min = h_total;
  let mut y_max = 0usize;
  for row in 0..h_total {
    let base = row * w_total;
    if a.pixels[base..base + w_total].iter().any(|&px| px != 0) {
      y_min = y_min.min(row);
      y_max = y_max.max(row);
    }
  }
  let (y_off, h_tight) = if y_min <= y_max {
    (y_min as u16, (y_max - y_min + 1) as u16)
  } else {
    (0u16, 0u16)
  };

  // Header (6 bytes), then glyph streams.
  let mut out = Vec::with_capacity(6);
  out.extend_from_slice(&a.width.to_le_bytes());
  out.extend_from_slice(&h_tight.to_le_bytes());
  out.extend_from_slice(&y_off.to_le_bytes());

  let mut next_offs: Vec<u16> = Vec::with_capacity(glyphs.len());

  if h_tight == 0 {
    return (out, next_offs, a.width, h_tight, y_off);
  }

  let y_start = y_off as usize;
  let y_end_excl = y_start + h_tight as usize;

  for g in glyphs {
    let gx = g.x as usize;
    let gw = g.w as usize;
    let bytes_per_row = (gw + 1) / 2;

    // 1) Pack this glyph's band into L4 BYTES (2 px/byte), row-major.
    let mut l4_bytes: Vec<u8> = Vec::with_capacity(bytes_per_row * (h_tight as usize));
    for row in y_start..y_end_excl {
      let base = row * w_total;
      let mut x = 0usize;
      while x < gw {
        let l = {
          let src_x = gx + x;
          if src_x < w_total {
            quant_l4_round(a.pixels[base + src_x])
          } else {
            0
          }
        } & 0x0F;
        let r = if x + 1 < gw {
          let src_x = gx + x + 1;
          if src_x < w_total {
            quant_l4_round(a.pixels[base + src_x])
          } else {
            0
          }
        } else {
          0
        } & 0x0F;
        l4_bytes.push((r << 4) | l);
        x += 2;
      }
    }

    // 2) RLE-encode BYTES.
    let rle = rle_encode_bytes(&l4_bytes);

    // 3) Append; record size as u16.
    let sz = rle.len();
    if sz > u16::MAX as usize {
      panic!("glyph RLE exceeds u16: {} bytes at x={}, w={}", sz, g.x, g.w);
    }
    next_offs.push(sz as u16);
    out.extend_from_slice(&rle);
  }

  (out, next_offs, a.width, h_tight, y_off)
}

/// RLE over BYTES.
/// - Runs for >=3 bytes (up to 128): emit 0x80..0xFF then 1 value byte.
/// - Otherwise emit literals of up to 128 bytes: 0x00..0x7F then `len` raw bytes.
fn rle_encode_bytes(src: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(src.len());
  let mut i = 0usize;
  while i < src.len() {
    // Try a run
    let v = src[i];
    let mut run = 1usize;
    while i + run < src.len() && src[i + run] == v && run < 128 {
      run += 1;
    }
    if run >= 3 {
      out.push(0x80 | ((run as u8) - 1));
      out.push(v);
      i += run;
      continue;
    }

    // Literal: gather up to 128 bytes or until a run>=3 would start.
    let lit_start = i;
    let mut lit_len = 1usize;
    i += 1;
    while i < src.len() {
      // Look-ahead for next run
      let vv = src[i];
      let mut r = 1usize;
      while i + r < src.len() && src[i + r] == vv && r < 128 {
        r += 1;
      }
      if r >= 3 || lit_len >= 128 {
        break;
      }
      lit_len += 1;
      i += 1;
    }
    out.push((lit_len as u8) - 1);
    out.extend_from_slice(&src[lit_start..lit_start + lit_len]);
  }
  out
}

// ──────────────────────────────────────────────────────────────────────────────
// binary tables
// ──────────────────────────────────────────────────────────────────────────────

struct GlyphTable {
  bytes: Vec<u8>,
  count: u16,
}
impl GlyphTable {
  fn from_slice_with_next(glyphs: &[Glyph], next_offs: &[u16]) -> anyhow::Result<Self> {
    if glyphs.is_empty() {
      anyhow::bail!("glyphs are required");
    }
    if glyphs.len() != next_offs.len() {
      anyhow::bail!("next_offs len {} != glyphs {}", next_offs.len(), glyphs.len());
    }
    let count: u16 = glyphs
      .len()
      .try_into()
      .map_err(|_| anyhow::anyhow!("too many glyphs"))?;
    let mut bytes = Vec::with_capacity(glyphs.len() * 7);
    for (g, off) in glyphs.iter().zip(next_offs.iter().copied()) {
      bytes.extend_from_slice(&glyph_to_bytes_with_next(g, off));
    }
    Ok(Self { bytes, count })
  }
}

struct KerningTable {
  bytes: Vec<u8>,
  count: u32,
}
impl KerningTable {
  fn from_slice(kerning: &[Kerning]) -> anyhow::Result<Self> {
    if kerning.is_empty() {
      return Ok(Self { bytes: Vec::new(), count: 0 });
    }
    let packed: anyhow::Result<Vec<[u8; 7]>> = kerning.iter().map(Kerning::to_bytes).collect();
    let vec = packed?;
    Ok(Self { bytes: vec.into_iter().flatten().collect(), count: kerning.len() as u32 })
  }
}

struct CharsetSegments {
  items: Vec<(u32, u16, u16)>,
  count: u16,
}
impl CharsetSegments {
  fn build(charset: &[CharsetRange], glyph_len: usize) -> anyhow::Result<Self> {
    if charset.is_empty() {
      anyhow::bail!("charset is required");
    }
    let mut items = Vec::with_capacity(charset.len());
    let mut base: u16 = 0;
    for (i, r) in charset.iter().enumerate() {
      let (seg, next) = r
        .segment_with_base(base)
        .map_err(|e| anyhow::anyhow!("charset[{i}]: {e}"))?;
      items.push(seg);
      base = next;
    }
    if base as usize != glyph_len {
      anyhow::bail!("charset codepoints {} != glyphs {}", base, glyph_len);
    }
    let count: u16 = items
      .len()
      .try_into()
      .map_err(|_| anyhow::anyhow!("too many charset segments"))?;
    Ok(Self { items, count })
  }
  fn write_to(&self, out: &mut Vec<u8>) {
    for (start_cp, len, base) in &self.items {
      push_u24(out, *start_cp);
      out.extend_from_slice(&len.to_le_bytes());
      out.extend_from_slice(&base.to_le_bytes());
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// assemble (public façade)
// ──────────────────────────────────────────────────────────────────────────────

pub fn assemble(meta: Metadata, atlas: Atlas) -> anyhow::Result<Vec<u8>> {
  if meta.line_height == 0 {
    anyhow::bail!("line_height must be non-zero");
  }

  // Build per-glyph RLE blob + per-glyph sizes.
  let (atlas_blob, next_offs, _w_total, _h_tight, _y_off) = build_pgrle_blob(&atlas, &meta.glyphs);

  let glyphs = GlyphTable::from_slice_with_next(&meta.glyphs, &next_offs)?;
  let kern = KerningTable::from_slice(&meta.kerning)?;
  let segs = CharsetSegments::build(&meta.charset, meta.glyphs.len())?;

  // layout
  let segments_len = segs.items.len() * 7;
  let header_len = HDR_PREFIX_LEN + segments_len;

  let glyphs_off = header_len as u32;
  let glyphs_len = glyphs.bytes.len() as u32;

  let atlas_off = (header_len + glyphs.bytes.len()) as u32;
  let atlas_len = atlas_blob.len() as u32;

  let mut out = Vec::with_capacity(header_len + glyphs.bytes.len() + atlas_blob.len() + kern.bytes.len());

  // header
  out.extend_from_slice(MAGIC);
  out.push(VERSION);
  out.push(MFNT_FLAG_ATLAS_PGRLE);
  out.extend_from_slice(&meta.line_height.to_le_bytes());
  out.extend_from_slice(&meta.ascent.to_le_bytes());
  out.extend_from_slice(&meta.descent.to_le_bytes());
  out.extend_from_slice(&glyphs.count.to_le_bytes());
  out.extend_from_slice(&glyphs_off.to_le_bytes());
  out.extend_from_slice(&glyphs_len.to_le_bytes());
  out.extend_from_slice(&atlas_off.to_le_bytes());
  out.extend_from_slice(&atlas_len.to_le_bytes());

  // backpatch slots
  let off_total_len = out.len();
  out.extend_from_slice(&0u32.to_le_bytes());
  let off_kern_off = out.len();
  out.extend_from_slice(&0u32.to_le_bytes());
  let off_kern_cnt = out.len();
  out.extend_from_slice(&0u32.to_le_bytes());

  out.extend_from_slice(&segs.count.to_le_bytes());
  segs.write_to(&mut out);

  // tables
  out.extend_from_slice(&glyphs.bytes);
  out.extend_from_slice(&atlas_blob);

  // kerning tail
  let (kern_off, kern_cnt) = if kern.bytes.is_empty() {
    (0, 0)
  } else {
    let off = out.len() as u32;
    out.extend_from_slice(&kern.bytes);
    (off, kern.count)
  };

  // backpatch
  let total_len = out.len() as u32;
  out[off_total_len..off_total_len + 4].copy_from_slice(&total_len.to_le_bytes());
  out[off_kern_off..off_kern_off + 4].copy_from_slice(&kern_off.to_le_bytes());
  out[off_kern_cnt..off_kern_cnt + 4].copy_from_slice(&kern_cnt.to_le_bytes());

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
fn put_u24(dst: &mut [u8], v: u32) {
  dst[0] = (v & 0xFF) as u8;
  dst[1] = ((v >> 8) & 0xFF) as u8;
  dst[2] = ((v >> 16) & 0xFF) as u8;
}
#[inline]
fn push_u24(buf: &mut Vec<u8>, v: u32) {
  buf.extend_from_slice(&[(v & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, ((v >> 16) & 0xFF) as u8]);
}
