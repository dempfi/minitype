//! MiniType font: compact, MCU-friendly container + linear L4 atlas.
//!
//! Atlas blob layout (from `Atlas::build_linear_l4`):
//!   u16 w, u16 h_tight, u16 y_off,
//!   rows (L4, 2px/byte; contiguous rows for y ∈ [y_off .. y_off + h_tight))
//!
//! MFNT layout:
//!   "MFNT", u8 version=1, u8 flags=0,
//!   u16 line_height, i16 ascent, i16 descent,
//!   u16 glyph_count,
//!   u32 glyphs_off, u32 glyphs_len,
//!   u32 atlas_off,  u32 atlas_len,
//!   u32 total_len (backpatch),
//!   u32 kerning_off (backpatch), u32 kerning_cnt (backpatch),
//!   u16 segment_count,
//!   segment_count × { u24 start_cp, u16 len, u16 base },
//!   glyph table (u16 x, u8 w, i8 advance, i8 left),   // 5 bytes/record
//!   atlas blob,
//!   kerning table (u24 left_cp, u24 right_cp, i8 adj).

use core::convert::TryInto;

// ──────────────────────────────────────────────────────────────────────────────
// constants & helpers
// ──────────────────────────────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"MFNT";
const VERSION: u8 = 1;

/// MFNT header bytes before charset segments start (fixed-size prefix).
const HDR_PREFIX_LEN: usize = 0x2C;

/// L4 encoding: 2 pixels per byte (low nibble = left pixel).
const L4_PIXELS_PER_BYTE: usize = 2;

#[inline]
fn quant_l4_round(v: u8) -> u8 {
  // nearest multiple of 16, clamped to 15
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

  /// Encodes: u16 x, u8 w, i8 advance, i8 left
  #[inline]
  pub fn to_bytes(&self) -> [u8; 5] {
    let mut out = [0u8; 5];
    out[..2].copy_from_slice(&self.x.to_le_bytes());
    out[2] = self.w;
    out[3] = self.advance_or_width() as u8; // i8 two's complement
    out[4] = self.left_bearing_or_zero() as u8; // i8 two's complement
    out
  }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Kerning {
  pub left: String,
  pub right: String,
  pub adj: i8,
}

impl Kerning {
  /// Encodes: u24 left, u24 right, i8 adj.
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

  /// Returns ((start_cp, len, base), next_base).
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
// atlas building
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

  /// Build linear L4 atlas; quantize with nearest (round/16). Serialized blob (w,h_tight,y_off, rows)
  pub fn build(&self) -> Vec<u8> {
    let w = self.width as usize;
    let h = self.height as usize;

    // --- Tight vertical bounds (any nonzero pixel in A8 source) --------------
    let mut y_min = h; // first non-empty row (inclusive)
    let mut y_max = 0usize; // last  non-empty row (inclusive)
    for row in 0..h {
      let base = row * w;
      if self.pixels[base..base + w].iter().any(|&px| px != 0) {
        y_min = y_min.min(row);
        y_max = y_max.max(row);
      }
    }

    // Handle all-black atlas gracefully.
    let (y_off, h_tight) = if y_min <= y_max {
      (y_min as u16, (y_max - y_min + 1) as u16)
    } else {
      (0u16, 0u16)
    };

    // Header (6 bytes) + tight rows
    let rows_bytes_per_row = (w + 1) / L4_PIXELS_PER_BYTE;
    let tight_rows_len = (h_tight as usize) * rows_bytes_per_row;
    let mut blob = Vec::with_capacity(6 + tight_rows_len);

    // Header: width, h_tight, y_off.
    blob.extend_from_slice(&self.width.to_le_bytes());
    blob.extend_from_slice(&h_tight.to_le_bytes());
    blob.extend_from_slice(&y_off.to_le_bytes());

    // Encode ONLY the contiguous tight band using linear L4.
    if h_tight > 0 {
      let y_start = y_off as usize;
      let y_end_excl = y_start + h_tight as usize;

      for row in y_start..y_end_excl {
        let base = row * w;
        let mut packed_row = Vec::with_capacity(rows_bytes_per_row);

        let mut x = 0usize;
        while x < w {
          let v_l = self.pixels[base + x];
          let il = quant_l4_round(v_l) & 0x0F;

          let ir = if x + 1 < w {
            let v_r = self.pixels[base + x + 1];
            quant_l4_round(v_r) & 0x0F
          } else {
            0
          };

          packed_row.push((ir << 4) | il);
          x += 2;
        }

        blob.extend_from_slice(&packed_row);
      }
    }

    blob
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// binary tables
// ──────────────────────────────────────────────────────────────────────────────

struct GlyphTable {
  bytes: Vec<u8>,
  count: u16,
}
impl GlyphTable {
  fn from_slice(glyphs: &[Glyph]) -> anyhow::Result<Self> {
    if glyphs.is_empty() {
      anyhow::bail!("glyphs are required");
    }
    let count: u16 = glyphs
      .len()
      .try_into()
      .map_err(|_| anyhow::anyhow!("too many glyphs"))?;
    Ok(Self { bytes: glyphs.iter().flat_map(Glyph::to_bytes).collect(), count })
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

  let atlas_blob = atlas.build();

  let glyphs = GlyphTable::from_slice(&meta.glyphs)?;
  let kern = KerningTable::from_slice(&meta.kerning)?;
  let segs = CharsetSegments::build(&meta.charset, meta.glyphs.len())?;

  // ── layout calculations ─────────────────────────────────────────
  let segments_len = segs.items.len() * 7;
  let header_len = HDR_PREFIX_LEN + segments_len;

  let glyphs_off = header_len as u32;
  let glyphs_len = glyphs.bytes.len() as u32;

  let atlas_off = (header_len + glyphs.bytes.len()) as u32;
  let atlas_len = atlas_blob.len() as u32;

  let mut out = Vec::with_capacity(header_len + glyphs.bytes.len() + atlas_blob.len() + kern.bytes.len());

  // ── header ──────────────────────────────────────────────────────
  out.extend_from_slice(MAGIC);
  out.push(VERSION);
  out.push(0); // flags
  out.extend_from_slice(&meta.line_height.to_le_bytes());
  out.extend_from_slice(&meta.ascent.to_le_bytes());
  out.extend_from_slice(&meta.descent.to_le_bytes());
  out.extend_from_slice(&glyphs.count.to_le_bytes());
  out.extend_from_slice(&glyphs_off.to_le_bytes());
  out.extend_from_slice(&glyphs_len.to_le_bytes());
  out.extend_from_slice(&atlas_off.to_le_bytes());
  out.extend_from_slice(&atlas_len.to_le_bytes());

  // Record backpatch positions.
  let off_total_len = out.len();
  out.extend_from_slice(&0u32.to_le_bytes()); // total_len (backpatch)
  let off_kern_off = out.len();
  out.extend_from_slice(&0u32.to_le_bytes()); // kerning_off (backpatch)
  let off_kern_cnt = out.len();
  out.extend_from_slice(&0u32.to_le_bytes()); // kerning_cnt (backpatch)

  out.extend_from_slice(&segs.count.to_le_bytes());

  // ── segments ────────────────────────────────────────────────────
  segs.write_to(&mut out);

  // ── tables ──────────────────────────────────────────────────────
  out.extend_from_slice(&glyphs.bytes);
  out.extend_from_slice(&atlas_blob);

  // Kerning tail (optional).
  let (kern_off, kern_cnt) = if kern.bytes.is_empty() {
    (0, 0)
  } else {
    let off = out.len() as u32;
    out.extend_from_slice(&kern.bytes);
    (off, kern.count)
  };

  // ── backpatch ───────────────────────────────────────────────────
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
  // Accept exactly one Unicode scalar; map it to its code point.
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
