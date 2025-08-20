//! MiniType font: compact, MCU-friendly container + *optimal* atlas packing.
//!
//! Atlas blob layout (from `Atlas::build_multi`):
//!   u16 w, u16 h_tight, u16 y_off,
//!   [u8;64] palettes (4×16; index 0 = black in each),
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
//!   glyph table (u16 x, u8 w, i8 advance, i8 left, u8 q),
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

/// Number of palettes and entries per palette in the atlas.
const PALETTE_COUNT: usize = 4;
const PALETTE_SIZE: usize = 16;

/// L4 encoding: 2 pixels per byte (low nibble = left pixel).
const L4_PIXELS_PER_BYTE: usize = 2;

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
  /// Quantization/dequantization hints byte:
  /// bits 0..1: palette id (0..3)
  /// bits 2..3: scale idx (0..3) -> step ≈ {8,12,16,24}
  /// bits 4..7: bias16 (0..15) -> bias = bias16*16
  #[serde(default)]
  pub q: Option<u8>,
}

impl Glyph {
  pub fn new(x: u16, w: u8, advance: Option<i8>, left: Option<i8>) -> Self {
    Self { x, w, advance, left, q: None }
  }
  #[inline]
  fn advance_or_width(&self) -> i8 {
    // Default advance = glyph width (common for monospaced/pixel fonts).
    self.advance.unwrap_or(self.w as i8)
  }
  #[inline]
  fn left_bearing_or_zero(&self) -> i8 {
    self.left.unwrap_or(0)
  }
  #[inline]
  fn q_or_zero(&self) -> u8 {
    self.q.unwrap_or(0)
  }

  /// Encodes: u16 x, u8 w, i8 advance, i8 left, u8 q.
  #[inline]
  pub fn to_bytes(&self) -> [u8; 6] {
    let mut out = [0u8; 6];
    out[..2].copy_from_slice(&self.x.to_le_bytes());
    out[2] = self.w;
    out[3] = self.advance_or_width() as u8; // i8 two's complement
    out[4] = self.left_bearing_or_zero() as u8; // i8 two's complement
    out[5] = self.q_or_zero();
    out
  }

  #[inline]
  pub fn with_q(&self, q: u8) -> Self {
    let mut g = self.clone();
    g.q = Some(q);
    g
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
  pub pixels: Vec<u8>, // L8 source
}

#[derive(Debug)]
pub struct AtlasBlob {
  pub qbytes: Vec<u8>, // per-glyph quantization bytes
  pub data: Vec<u8>,   // serialized blob (w,h_tight,y_off,palettes,rows)
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

  /// Build L4 atlas with 4 palettes; choose per-glyph `q` to minimize **decoded** MSE.
  /// This matches the reader’s fused affine-LS LUT, so “what you encode is what the
  /// MCU will approximately decode”.
  pub fn build_multi(&self, glyphs: &[Glyph]) -> AtlasBlob {
    let w = self.width as usize;
    let h = self.height as usize;

    // --- Pick palettes --------------------------------------------------------
    // p0: full mass, p1: dark-biased, p2: bright-biased, p3: uniform.
    let hist = self.histogram_256();
    let p0 = Self::palette16_from_hist(&hist, 1, 255);
    let p1 = Self::palette16_from_hist(&hist, 1, 128);
    let p2 = Self::palette16_from_hist(&hist, 128, 255);
    let p3 = Self::palette16_uniform();

    let mut palettes = [[0u8; PALETTE_SIZE]; PALETTE_COUNT];
    palettes[0] = p0;
    palettes[1] = p1;
    palettes[2] = p2;
    palettes[3] = p3;

    // Precompute nearest tables (byte → palette index nibble) for each palette.
    let nearest = Self::build_nearest_tables(&palettes);

    // Precompute fused decode LUTs for all 256 q-combos (pid×scale×bias) to speed search.
    let mut fused_luts = [[0u8; PALETTE_SIZE]; 256];
    for pid in 0u8..4 {
      for sc in 0u8..4 {
        for b16 in 0u8..16 {
          let idx = combo_idx(pid, sc, b16);
          fused_luts[idx] = Self::fused_lut_affine_ls(&palettes[pid as usize], b16, sc);
        }
      }
    }

    // Per-x palette map (filled with chosen per-glyph palette id).
    let mut palette_for_x = vec![0u8; w];

    // Outputs per glyph: chosen q byte and palette id.
    let mut q_bytes = Vec::with_capacity(glyphs.len());
    let mut glyph_pid = Vec::with_capacity(glyphs.len());

    // --- Choose best q for each glyph via histogram-based exact SSE -----------
    for g in glyphs {
      let gx = g.x as usize;
      let gw = g.w as usize;

      if gw == 0 {
        q_bytes.push(0);
        glyph_pid.push(0);
        continue;
      }

      // Histogram only over the glyph’s vertical strip [gx, gx+gw) across all rows.
      let ghist = Self::glyph_histogram_l8(&self.pixels, w, h, gx, gw);

      // All-zero glyph shortcut.
      if ghist.iter().skip(1).all(|&c| c == 0) {
        q_bytes.push(0);
        glyph_pid.push(0);
        continue;
      }

      let mut best_err: u64 = u64::MAX;
      let mut best_q: u8 = 0;
      let mut best_pid: u8 = 0;

      // Exhaustive search over palette id, scale idx, bias.
      for pid in 0u8..4 {
        let ntab = &nearest[pid as usize];
        for sc in 0u8..4 {
          for b16 in 0u8..16 {
            let lut = &fused_luts[combo_idx(pid, sc, b16)];
            let err = glyph_error_sse(&ghist, ntab, lut);
            if err < best_err {
              best_err = err;
              best_q = ((b16 & 0x0F) << 4) | ((sc & 0b11) << 2) | (pid & 0b11);
              best_pid = pid;
            }
          }
        }
      }

      q_bytes.push(best_q);
      glyph_pid.push(best_pid);

      // Mark chosen palette across x-span of this glyph (used during row packing).
      let x1 = gx + gw;
      for x in gx..x1 {
        palette_for_x[x] = best_pid;
      }
    }

    // --- Serialize atlas: w,h, palettes(4×16), rowmask, rows(L4) --------------

    // Compute tight vertical bounds in L8 source (any non-zero pixel counts).
    let mut y_min = h; // first non-empty row (inclusive)
    let mut y_max = 0usize; // last  non-empty row (inclusive)
    for row in 0..h {
      let base = row * w;
      // Fast scan: early-out as soon as we see a non-zero pixel.
      if self.pixels[base..base + w].iter().any(|&px| px != 0) {
        y_min = y_min.min(row);
        y_max = y_max.max(row);
      }
    }

    // Handle all-black atlas gracefully.
    let (y_off, h_tight) = if y_min <= y_max {
      (y_min as u16, (y_max - y_min + 1) as u16)
    } else {
      // No non-zero rows: empty payload (keep y_off=0, h_tight=0).
      (0u16, 0u16)
    };

    // Header (6 bytes) + palettes + tight rows
    let rows_bytes_per_row = (w + 1) / L4_PIXELS_PER_BYTE;
    let tight_rows_len = (h_tight as usize) * rows_bytes_per_row;
    let mut blob = Vec::with_capacity(6 + (PALETTE_COUNT * PALETTE_SIZE) + tight_rows_len);

    // Header: width, h_tight, y_off.
    blob.extend_from_slice(&self.width.to_le_bytes());
    blob.extend_from_slice(&h_tight.to_le_bytes());
    blob.extend_from_slice(&y_off.to_le_bytes());

    // Palettes (index 0 must be 0 for “transparent/black”).
    for pid in 0..PALETTE_COUNT {
      blob.extend_from_slice(&palettes[pid]);
    }

    // Encode ONLY the contiguous tight band.
    if h_tight > 0 {
      let y_start = y_off as usize;
      let y_end_excl = y_start + h_tight as usize;

      for row in y_start..y_end_excl {
        let base = row * w;
        let mut packed_row = Vec::with_capacity(rows_bytes_per_row);

        let mut x = 0usize;
        while x < w {
          // Left pixel → low nibble
          let pid_l = palette_for_x[x] as usize;
          let il = nearest[pid_l][self.pixels[base + x] as usize] & 0x0F;

          // Right pixel → high nibble (if present)
          let ir = if x + 1 < w {
            let pid_r = palette_for_x[x + 1] as usize;
            nearest[pid_r][self.pixels[base + x + 1] as usize] & 0x0F
          } else {
            0u8
          };

          packed_row.push((ir << 4) | il);
          x += 2;
        }

        blob.extend_from_slice(&packed_row);
      }
    }

    AtlasBlob { qbytes: q_bytes, data: blob }
  }

  // ── palette helpers ────────────────────────────────────────────

  #[inline]
  fn histogram_256(&self) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &v in &self.pixels {
      hist[v as usize] += 1;
    }
    hist
  }

  /// Build a 16-entry palette from histogram with a bias window [lo..hi].
  #[inline]
  fn palette16_from_hist(hist: &[u32; 256], lo: u8, hi: u8) -> [u8; PALETTE_SIZE] {
    // Collect nonzero frequencies in the window, excluding 0 (reserved).
    let mut freq: Vec<(u8, u32)> = (lo..=hi)
      .map(|v| (v, hist[v as usize]))
      .filter(|&(v, c)| v != 0 && c > 0)
      .collect();

    // Sort by descending frequency, then ascending value for stable picks.
    freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Take top values (up to 15), keeping them sorted & unique.
    let mut top: Vec<u8> = freq.into_iter().take(15).map(|(v, _)| v).collect();
    top.sort_unstable();
    top.dedup();

    // Backfill uniformly to reach 15 if needed, to ensure coverage in window.
    let mut i = 1;
    while top.len() < 15 {
      let v = ((i as u32 * (hi as u32 - lo as u32 + 1)) / 16) as u32 + lo as u32;
      let v = v.saturating_sub(1).min(255) as u8;
      if v != 0 && top.binary_search(&v).is_err() {
        top.push(v);
      }
      i += 1;
    }
    top.sort_unstable();

    let mut pal = [0u8; PALETTE_SIZE];
    for (idx, &v) in top.iter().enumerate() {
      pal[idx + 1] = v;
    }
    pal
  }

  #[inline]
  fn palette16_uniform() -> [u8; PALETTE_SIZE] {
    let mut pal = [0u8; PALETTE_SIZE];
    for i in 1..PALETTE_SIZE {
      pal[i] = (((i as u32) * 256) / 16).saturating_sub(1) as u8; // ≈ 16*i - 1
    }
    pal
  }

  #[inline]
  fn step_from_scale_idx(idx: u8) -> u16 {
    match idx & 0b11 {
      0 => 8,
      1 => 12,
      2 => 16,
      _ => 24,
    }
  }

  /// Build fused LUT via affine least-squares fit mapping palette entries p[n]
  /// to target ramp t[n] = clamp(bias + step*n, 255); monotonic enforced.
  fn fused_lut_affine_ls(pal: &[u8; PALETTE_SIZE], bias16: u8, scale_idx: u8) -> [u8; PALETTE_SIZE] {
    let bias = (bias16 as u16) * 16u16;
    let step = Self::step_from_scale_idx(scale_idx) as u16;

    const N: i32 = 15; // using entries 1..15
    let mut sum_p: i32 = 0;
    let mut sum_t: i32 = 0;
    let mut sum_p2: i32 = 0;
    let mut sum_pt: i32 = 0;

    // Accumulate LS terms over palette entries 1..15 (0 is reserved).
    for n in 1..16 {
      let p = pal[n] as i32;
      let t_raw = bias as i32 + (step as i32) * (n as i32);
      let t = t_raw.min(255);
      sum_p += p;
      sum_t += t;
      sum_p2 += p * p;
      sum_pt += p * t;
    }

    // Solve slope (s) and offset (o) in 16.16 fixed-point. Fallback to sane defaults.
    let denom = (N * sum_p2 - sum_p * sum_p) as i64;
    let (s_fp, o_fp) = if denom.abs() <= 1 {
      let maxp = pal[15] as i32;
      let s_fp = if maxp > 0 {
        (((step as i32) * 15) << 16) / maxp
      } else {
        1 << 16
      };
      ((s_fp), (bias as i32) << 16)
    } else {
      let num_s = (N as i64) * (sum_pt as i64) - (sum_p as i64) * (sum_t as i64);
      let s_fp = ((num_s << 16) / denom) as i32;
      let num_o = ((sum_t as i64) << 16) - (s_fp as i64) * (sum_p as i64);
      let o_fp = (num_o / (N as i64)) as i32;
      (s_fp, o_fp)
    };

    // Produce LUT with monotonicity enforced.
    let mut lut = [0u8; PALETTE_SIZE];
    lut[0] = 0;
    let mut prev: i32 = 0;
    for i in 1..PALETTE_SIZE {
      let p = pal[i] as i32;
      // s_fp (16.16) * p (int) + o_fp (16.16) → keep in 16.16, then >>16.
      let val_fp = o_fp + (((s_fp as i64) * (p as i64)) as i32);
      let mut v = (val_fp >> 16).clamp(0, 255);
      if v < prev {
        v = prev; // enforce monotonicity
      }
      prev = v;
      lut[i] = v as u8;
    }
    lut
  }

  /// 4×(256→nibble) nearest tables for each palette.
  fn build_nearest_tables(palettes: &[[u8; PALETTE_SIZE]; PALETTE_COUNT]) -> [[u8; 256]; PALETTE_COUNT] {
    let mut nearest = [[0u8; 256]; PALETTE_COUNT];
    for pid in 0..PALETTE_COUNT {
      for v in 0..=255u16 {
        let mut best_i = 0u8;
        let mut best_d = u16::MAX;
        for (i, &p) in palettes[pid].iter().enumerate() {
          let d = v.abs_diff(p as u16);
          if d < best_d || (d == best_d && (i as u8) < best_i) {
            best_d = d;
            best_i = i as u8;
          }
        }
        nearest[pid][v as usize] = best_i & 0x0F;
      }
    }
    nearest
  }

  /// Histogram over the glyph’s rectangle in the source L8 atlas.
  fn glyph_histogram_l8(pixels: &[u8], atlas_w: usize, atlas_h: usize, gx: usize, gw: usize) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for y in 0..atlas_h {
      let row = &pixels[y * atlas_w..y * atlas_w + atlas_w];
      for x in 0..gw {
        let v = row[gx + x];
        hist[v as usize] += 1;
      }
    }
    hist
  }
}

/// SSE given hist, nearest table for palette, and 16-entry fused LUT.
/// Note: v=0 contributes 0 error by construction (nearest=0, lut[0]=0).
#[inline]
fn glyph_error_sse(hist: &[u32; 256], nearest256: &[u8; 256], lut16: &[u8; PALETTE_SIZE]) -> u64 {
  let mut err: u64 = 0;
  for v in 1..256 {
    let c = hist[v] as u64;
    if c != 0 {
      let n = nearest256[v] as usize;
      let a = lut16[n] as i32;
      let d = (v as i32) - a;
      err += (d as i64 * d as i64) as u64 * c;
    }
  }
  err
}

#[inline]
fn combo_idx(pal_id: u8, scale_idx: u8, bias16: u8) -> usize {
  ((pal_id as usize) << 6) | ((scale_idx as usize) << 4) | (bias16 as usize)
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

  // Build atlas & compute per-glyph q bytes (optimal decoded MSE).
  let AtlasBlob { data: atlas_blob, qbytes } = atlas.build_multi(&meta.glyphs);

  // Enrich glyphs with chosen q.
  let enriched: Vec<Glyph> = meta
    .glyphs
    .iter()
    .zip(qbytes.into_iter())
    .map(|(g, q)| g.with_q(q))
    .collect();

  let glyphs = GlyphTable::from_slice(&enriched)?;
  let kern = KerningTable::from_slice(&meta.kerning)?;
  let segs = CharsetSegments::build(&meta.charset, enriched.len())?;

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
