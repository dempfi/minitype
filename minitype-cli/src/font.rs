/// Font container builder and serializer for MiniType atlas.
///
/// This module defines a compact, MCU‑friendly **font container** that bundles:
/// - the compressed **glyph atlas** (inner MiniType atlas produced by `atlas::compress`),
/// - **glyph metrics & mapping** (per‑glyph rectangles, layout metrics, and advance),
/// - global **font metrics** (line height, ascent, descent),
/// - optional **kerning pairs** (left/right glyph codepoints and adjustment),
/// - **charset segments** to map Unicode ranges to glyph indices,
///
/// # Binary container layout (little‑endian)
/// ```text
/// // Offsets are from the start of the container
/// 0x00 4  MAGIC = "MFNT"
/// 0x04 1  VERSION = 1
/// 0x05 1  FLAGS (Reserved = 0)
/// 0x06 2  line_height (u16)
/// 0x08 2  ascent (i16)
/// 0x0A 2  descent (i16)
/// 0x0C 2  glyph_count (u16)
/// 0x0E 4  glyph_table_offset (u32)
/// 0x12 4  glyph_table_len    (u32)   // bytes; each record is 4 bytes: u16 x, u8 w, i8 advance
/// 0x16 4  atlas_offset       (u32)
/// 0x1A 4  atlas_len          (u32)   // bytes; see atlas header below
/// 0x1E 4  total_len          (u32)
/// 0x22 4  kerning_offset     (u32)   // 0 if none
/// 0x26 4  kerning_count      (u32)   // number of pairs; each is 7 bytes (u24 left, u24 right, i8 adj)
/// 0x2A 2  charset_seg_count  (u16)
/// 0x2C .. charset_segments[charset_seg_count]  // each 7 bytes: u24 start, u16 len, u16 base
/// ...   .. glyph_table                        // 4 bytes per glyph: u16 x, u8 w, i8 advance
/// ...   .. atlas_blob
/// ...   .. kerning_pairs[kerning_count]       // optional, after atlas blob
/// ```
///
/// Atlas header (at atlas_offset):
///  - 0..2: width (u16 LE)
///  - 2..4: height (u16 LE)
///  - 4..20: palette[16] (u8)
///  - 20..(20 + ceil(h/8)): row mask
///  - ...: row payloads (for rows with mask bit=1)
use super::atlas::{build_palette_u8, compress};
use core::convert::TryInto;

#[inline]
fn push_u24_le(buf: &mut Vec<u8>, v: u32) {
  let b0 = (v & 0xFF) as u8;
  let b1 = ((v >> 8) & 0xFF) as u8;
  let b2 = ((v >> 16) & 0xFF) as u8;
  buf.extend_from_slice(&[b0, b1, b2]);
}

/// JSON schema for font metadata.
///
/// Example JSON:
/// ```json
/// {
///   "line_height": 16,
///   "ascent": 12,
///   "descent": -4,
///   "charset": [{"start":" ", "end":"~"}],
///   "glyphs": [ {"x":0, "w":5}, {"x":5, "w":6} ],
///   "kerning": [{"left":"A","right":"V","adj":-2}]
/// }
/// ```
/// Note: the number of glyphs must equal the total codepoints in the charset ranges.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Font {
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
}

impl Glyph {
  #[inline]
  fn resolved_advance(&self) -> i8 {
    self.advance.unwrap_or(self.w as i8)
  }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Kerning {
  pub left: String,
  pub right: String,
  pub adj: i8,
}

impl Kerning {
  fn left_cp(&self) -> anyhow::Result<u32> {
    let mut it = self.left.chars();
    let c = it.next().ok_or_else(|| anyhow::anyhow!("kerning 'left' is empty"))?;
    if it.next().is_some() {
      return Err(anyhow::anyhow!("kerning 'left' must be exactly one Unicode scalar"));
    }
    Ok(c as u32)
  }
  fn right_cp(&self) -> anyhow::Result<u32> {
    let mut it = self.right.chars();
    let c = it.next().ok_or_else(|| anyhow::anyhow!("kerning 'right' is empty"))?;
    if it.next().is_some() {
      return Err(anyhow::anyhow!("kerning 'right' must be exactly one Unicode scalar"));
    }
    Ok(c as u32)
  }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CharsetRange {
  pub start: String,
  pub end: String,
}

impl CharsetRange {
  fn start_cp(&self) -> anyhow::Result<u32> {
    single_scalar(&self.start, "charset.start")
  }
  fn end_cp(&self) -> anyhow::Result<u32> {
    single_scalar(&self.end, "charset.end")
  }
}

fn single_scalar(s: &str, what: &str) -> anyhow::Result<u32> {
  let mut it = s.chars();
  let c = it.next().ok_or_else(|| anyhow::anyhow!("{what} is empty"))?;
  if it.next().is_some() {
    return Err(anyhow::anyhow!("{what} must be exactly one Unicode scalar"));
  }
  Ok(c as u32)
}

/// Quantize an L8 image using the atlas palette strategy and return the post-quantized L8 buffer.
#[inline]
fn quantize_l8_with_atlas_palette(pixels: &[u8]) -> Vec<u8> {
  let (palette, map_idx) = build_palette_u8(pixels);
  let mut out = Vec::with_capacity(pixels.len());
  for &v in pixels {
    out.push(palette[map_idx[v as usize] as usize]);
  }
  out
}

/// Core builder: takes L8 coverage pixels and Font meta, compresses the atlas,
/// and serializes the MFNT container.
pub fn build_font_from_l8_and_meta(l8: &[u8], width: u16, height: u16, meta: &Font) -> anyhow::Result<Vec<u8>> {
  let atlas_blob = compress(l8, width, height);

  if meta.charset.is_empty() {
    return Err(anyhow::anyhow!("charset is required"));
  }

  let mut glyphs_info: Vec<(u16, u8, i8)> = Vec::with_capacity(meta.glyphs.len());
  for g in &meta.glyphs {
    glyphs_info.push((g.x, g.w, g.resolved_advance()));
  }

  let glyph_count: u16 = glyphs_info
    .len()
    .try_into()
    .map_err(|_| anyhow::anyhow!("too many glyphs"))?;

  let segments = build_charset_segments(&meta.charset, meta.glyphs.len())?;

  // Fixed header is 0x2C bytes, then charset segments (7 bytes each)
  let fixed_header_len: usize = 0x2C; // 44 bytes
  let header_len: usize = fixed_header_len + segments.len() * 7;
  let glyph_rec_len: usize = 4;
  let glyph_table_len = glyph_rec_len * (glyph_count as usize);
  let glyph_table_offset: u32 = header_len as u32;
  let atlas_offset: u32 = (header_len + glyph_table_len) as u32;
  let atlas_len: u32 = atlas_blob.len() as u32;

  // We'll append kerning block after atlas blob if any.
  // kerning_offset, kerning_count are placeholders for now.
  let mut out = Vec::with_capacity(header_len + glyph_table_len + atlas_blob.len() + meta.kerning.len() * 7);

  // Header
  out.extend_from_slice(b"MFNT"); // MAGIC
  out.push(1); // VERSION
  out.push(0); // FLAGS
  out.extend_from_slice(&meta.line_height.to_le_bytes());
  out.extend_from_slice(&meta.ascent.to_le_bytes());
  out.extend_from_slice(&meta.descent.to_le_bytes());
  out.extend_from_slice(&glyph_count.to_le_bytes());
  out.extend_from_slice(&glyph_table_offset.to_le_bytes());
  out.extend_from_slice(&(glyph_table_len as u32).to_le_bytes());
  out.extend_from_slice(&atlas_offset.to_le_bytes());
  out.extend_from_slice(&atlas_len.to_le_bytes());
  out.extend_from_slice(&(0u32).to_le_bytes()); // total_len placeholder
  out.extend_from_slice(&(0u32).to_le_bytes()); // kerning_offset placeholder
  out.extend_from_slice(&(0u32).to_le_bytes()); // kerning_count placeholder

  let charset_seg_count_u16: u16 = segments
    .len()
    .try_into()
    .map_err(|_| anyhow::anyhow!("too many charset segments"))?;

  out.extend_from_slice(&charset_seg_count_u16.to_le_bytes()); // charset_seg_count (u16)

  // Charset segments immediately after header
  for (start_cp, len, glyph_base) in &segments {
    push_u24_le(&mut out, *start_cp);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&glyph_base.to_le_bytes());
  }

  // Glyph table (always charset mode)
  for (x, w, adv) in &glyphs_info {
    out.extend_from_slice(&x.to_le_bytes());
    out.push(*w);
    out.push(*adv as u8);
  }

  // Atlas blob
  out.extend_from_slice(&atlas_blob);

  // Kerning block (optional)
  let kerning_offset: u32;
  let kerning_count: u32;
  if !meta.kerning.is_empty() {
    kerning_offset = out.len() as u32;
    kerning_count = meta.kerning.len() as u32;
    for k in &meta.kerning {
      let left_cp = k.left_cp()?;
      let right_cp = k.right_cp()?;
      push_u24_le(&mut out, left_cp);
      push_u24_le(&mut out, right_cp);
      out.push(k.adj as u8);
    }
  } else {
    kerning_offset = 0;
    kerning_count = 0;
  }

  // total_len at 0x1E
  let total_len = out.len() as u32;
  out[0x1E..0x22].copy_from_slice(&total_len.to_le_bytes());

  // kerning_offset at 0x22
  out[0x22..0x26].copy_from_slice(&kerning_offset.to_le_bytes());

  // kerning_count at 0x26
  out[0x26..0x2A].copy_from_slice(&kerning_count.to_le_bytes());

  Ok(out)
}

/// Validate and build charset segments from JSON charset ranges and glyph count.
/// Returns Vec<(start_cp: u32, len: u16, glyph_base: u16)>.
/// Ensures start <= end, length fits u16, and total glyphs fit u16.
fn build_charset_segments(charset: &[CharsetRange], glyph_count: usize) -> anyhow::Result<Vec<(u32, u16, u16)>> {
  let mut segments = Vec::with_capacity(charset.len());
  let mut glyph_base: u16 = 0;

  for (i, range) in charset.iter().enumerate() {
    let start_cp = range.start_cp()?;
    let end_cp = range.end_cp()?;
    if end_cp < start_cp {
      return Err(anyhow::anyhow!(
        "charset range {} end codepoint {:#X} is less than start {:#X}",
        i,
        end_cp,
        start_cp
      ));
    }
    let len = end_cp - start_cp + 1;
    if len > u16::MAX as u32 {
      return Err(anyhow::anyhow!("charset range {} length {} exceeds u16 max", i, len));
    }
    let len_u16 = len as u16;
    if (glyph_base as usize) + (len_u16 as usize) > u16::MAX as usize {
      return Err(anyhow::anyhow!("cumulative glyph_base + length exceeds u16 max at range {}", i));
    }
    segments.push((start_cp, len_u16, glyph_base));
    glyph_base = glyph_base
      .checked_add(len_u16)
      .ok_or_else(|| anyhow::anyhow!("glyph_base overflow after adding range {}", i))?;
  }

  if glyph_base as usize != glyph_count {
    return Err(anyhow::anyhow!(
      "total codepoints in charset ({}) does not match glyph count ({})",
      glyph_base,
      glyph_count,
    ));
  }

  Ok(segments)
}

/// ---
/// TTF -> MFNT one-shot builder
///
/// This module renders a TrueType/OpenType font at a fixed pixel size into a
/// single-row grayscale atlas, collects glyph advances and optional kerning,
/// and then invokes `build_font_from_png_and_json` to produce the final MFNT
/// container bytes.
pub mod ttfgen {
  use super::{CharsetRange, Font, Glyph};
  use anyhow::anyhow;
  use image::ImageEncoder as _;
  use swash::scale::{Render, ScaleContext, Source};
  use swash::zeno::Format;
  use swash::{FontRef, NormalizedCoord};

  /// Build a MFNT container and sidecar artifacts (L8 PNG atlas + JSON meta)
  /// directly from a TTF/OTF bytestream.
  /// Returns (MFNT_bytes, png_l8_bytes, json_meta_text).
  pub fn build_from_ttf_with_artifacts(
    ttf: &[u8],
    px: f32,
    ranges: &[(char, char)],
  ) -> anyhow::Result<(Vec<u8>, Vec<u8>, String)> {
    // 1) Load font (Swash)
    let font = FontRef::from_index(ttf, 0).ok_or_else(|| anyhow!("failed to parse TTF font"))?;

    // charset
    let mut charset: Vec<char> = Vec::new();
    for &(start, end) in ranges {
      if end < start {
        return Err(anyhow!("range end < start: {:?}-{:?}", start, end));
      }
      let mut c = start as u32;
      let end_u = end as u32;
      while c <= end_u {
        if let Some(ch) = char::from_u32(c) {
          charset.push(ch);
        }
        c += 1;
      }
    }
    if charset.is_empty() {
      return Err(anyhow!("charset empty"));
    }

    // 2) Vertical metrics from font units → pixels using Swash's scaled metrics
    let m_scaled = font.metrics(&[] as &[NormalizedCoord]).scale(px);
    let ascent = m_scaled.ascent; // distance baseline → top (px)
    let descent = m_scaled.descent; // distance baseline → bottom (px)
    let line_gap = m_scaled.leading; // recommended additional spacing (px)
    // Total baseline-to-baseline line height
    let line_height = (ascent + descent + line_gap).ceil().max(1.0) as u16;

    // 3) Prepare scaled glyph metrics for advances
    let gmetrics = font.glyph_metrics(&[] as &[NormalizedCoord]).scale(px);

    // 3) Rasterize with Swash (using outline-based grayscale, not bitmap strikes)
    let mut glyphs_vec: Vec<Glyph> = Vec::with_capacity(charset.len());
    let mut x_cursor: u32 = 0;
    // x, left, top, w, h, bitmap
    let mut bitmaps: Vec<(u16, i32, i32, i32, i32, Vec<u8>)> = Vec::with_capacity(charset.len());

    // Swash: grayscale (alpha) from outline at requested pixel size
    let outline = [Source::Outline];
    let mut render = Render::new(&outline);
    render.format(Format::Alpha);
    let mut scale_ctx = ScaleContext::new();
    let mut scaler = scale_ctx.builder(font).size(px).hint(true).build();

    for &ch in charset.iter() {
      let gid = font.charmap().map(ch);
      if let Some(img) = render.render(&mut scaler, gid) {
        let place = img.placement;
        let w = place.width.max(0) as i32;
        let h = place.height.max(0) as i32;
        let left = place.left as i32;
        let top = place.top as i32;

        let adv_px = gmetrics.advance_width(gid);
        let eff_adv_px = adv_px - (left as f32);

        let mut gb = vec![0u8; (w.max(0) as usize) * (h.max(0) as usize)];
        if w > 0 && h > 0 {
          gb.copy_from_slice(&img.data);
        }

        let gw = w.clamp(0, 255) as u8;
        glyphs_vec.push(Glyph { x: x_cursor as u16, w: gw, advance: Some(saturate_i8(eff_adv_px)) });

        bitmaps.push((x_cursor as u16, left, top, w, h, gb));
        x_cursor = x_cursor.saturating_add(w.max(0) as u32);
      } else {
        glyphs_vec.push(Glyph { x: x_cursor as u16, w: 0, advance: Some(0) });
      }
    }

    let atlas_w: u16 = (x_cursor as u16).max(1);
    // Baseline measured from the top of the atlas; height must include ascent + descent
    let baseline_y = ascent.ceil() as i32;
    let atlas_h = (ascent + descent).ceil().max(1.0) as u16;

    // 4) Compose the single-row L8 atlas using placement relative to baseline
    let mut l8 = vec![0u8; atlas_w as usize * atlas_h as usize];
    for (x, _left, top, w, h, bm) in &bitmaps {
      if *w <= 0 || *h <= 0 {
        continue;
      }
      let dst_top = baseline_y - *top; // baseline origin (Y grows downward)
      let dst_left = *x as i32; // packed by bitmap-left; no bearing here
      for row in 0..*h {
        let dst_y = dst_top + row;
        if dst_y < 0 || dst_y >= atlas_h as i32 {
          continue;
        }
        let dst_x = dst_left as usize;
        let row_w = *w as usize;
        let src_row_off = (row as usize) * row_w;
        let remaining = (atlas_w as usize).saturating_sub(dst_x);
        let copy_w = core::cmp::min(row_w, remaining);
        if copy_w > 0 {
          let src_off = src_row_off;
          let dst_off = (dst_y as usize) * (atlas_w as usize) + dst_x;
          l8[dst_off..dst_off + copy_w].copy_from_slice(&bm[src_off..src_off + copy_w]);
        }
      }
    }
    // 5) Quantize the L8 atlas using the same palette strategy as compressor for preview
    let l8_quant_preview = super::quantize_l8_with_atlas_palette(&l8);

    // 6) Build JSON meta
    let charset_json: Vec<CharsetRange> = ranges
      .iter()
      .map(|(s, e)| CharsetRange { start: s.to_string(), end: e.to_string() })
      .collect();

    let meta = Font {
      line_height,
      ascent: ascent.round() as i16,
      descent: descent.round() as i16,
      glyphs: glyphs_vec,
      kerning: Vec::new(), // can add shaping-based kerning later
      charset: charset_json,
    };
    let json_text = serde_json::to_string_pretty(&meta)?;

    // 7) Encode PNG atlas (post-quantization L8)
    let png = encode_l8_png(&l8_quant_preview, atlas_w, atlas_h)?;

    // 8) Build MFNT bytes via the core path (bake quantized atlas)
    let font = super::build_font_from_l8_and_meta(&l8, atlas_w, atlas_h, &meta)?;
    Ok((font, png, json_text))
  }

  /// Encode an L8 grayscale image to PNG bytes.
  fn encode_l8_png(pixels: &[u8], w: u16, h: u16) -> anyhow::Result<Vec<u8>> {
    use image::{ExtendedColorType, codecs::png::PngEncoder};
    let mut out = Vec::new();
    let enc = PngEncoder::new(&mut out);
    enc.write_image(pixels, w as u32, h as u32, ExtendedColorType::L8)?;
    Ok(out)
  }

  #[inline]
  pub(super) fn saturate_i8(v: f32) -> i8 {
    let r = v.round();
    if r < i8::MIN as f32 {
      i8::MIN
    } else if r > i8::MAX as f32 {
      i8::MAX
    } else {
      r as i8
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
  }
  fn i16_le(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([b[off], b[off + 1]])
  }
  fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
  }
  fn u24_le(b: &[u8], off: usize) -> u32 {
    (b[off] as u32) | ((b[off + 1] as u32) << 8) | ((b[off + 2] as u32) << 16)
  }

  fn make_meta_single_segment() -> Font {
    Font {
      line_height: 10,
      ascent: 7,
      descent: -3,
      // 3 glyphs total (A..C)
      glyphs: vec![
        Glyph { x: 0, w: 3, advance: Some(3) },
        Glyph { x: 3, w: 2, advance: Some(2) },
        Glyph { x: 5, w: 4, advance: Some(4) },
      ],
      kerning: vec![Kerning { left: "A".into(), right: "B".into(), adj: -1 }],
      charset: vec![CharsetRange { start: "A".into(), end: "C".into() }],
    }
  }

  #[test]
  fn basic_layout_single_segment() {
    let meta = make_meta_single_segment();
    // Atlas: width = sum of glyph widths = 9, height = 5
    let (w, h) = (9u16, 5u16);
    let mut l8 = vec![0u8; (w as usize) * (h as usize)];
    // Put a simple pattern so compressor has non-zero content
    for y in 0..h as usize {
      for x in 0..w as usize {
        l8[y * (w as usize) + x] = ((x + y) % 16) as u8 * 16 / 15; // small gradient in 0..15 scaled
      }
    }

    let bytes = build_font_from_l8_and_meta(&l8, w, h, &meta).expect("builder ok");

    // --- Fixed header checks ---
    assert_eq!(&bytes[0..4], b"MFNT");
    assert_eq!(bytes[4], 1); // VERSION
    assert_eq!(bytes[5], 0); // FLAGS
    assert_eq!(u16_le(&bytes, 0x06), meta.line_height);
    assert_eq!(i16_le(&bytes, 0x08), meta.ascent);
    assert_eq!(i16_le(&bytes, 0x0A), meta.descent);

    let glyph_count = u16_le(&bytes, 0x0C);
    assert_eq!(glyph_count, 3);

    let glyph_table_offset = u32_le(&bytes, 0x0E) as usize;
    let glyph_table_len = u32_le(&bytes, 0x12) as usize;
    let atlas_offset = u32_le(&bytes, 0x16) as usize;
    let atlas_len = u32_le(&bytes, 0x1A) as usize;
    let total_len = u32_le(&bytes, 0x1E) as usize;
    let kerning_offset = u32_le(&bytes, 0x22) as usize;
    let kerning_count = u32_le(&bytes, 0x26) as usize;
    let charset_seg_count = u16_le(&bytes, 0x2A) as usize;

    // Header size = 0x2C + 7 * charset_seg_count
    let header_len = 0x2C + 7 * charset_seg_count;
    assert_eq!(charset_seg_count, 1);

    assert_eq!(glyph_table_offset, header_len);
    assert_eq!(glyph_table_len, (glyph_count as usize) * 4);
    assert_eq!(atlas_offset, header_len + glyph_table_len);

    // Charset segment contents immediately after fixed header
    let seg_off = 0x2C;
    let seg_start = u24_le(&bytes, seg_off);
    let seg_len = u16_le(&bytes, seg_off + 3);
    let seg_base = u16_le(&bytes, seg_off + 5);
    assert_eq!(seg_start, 'A' as u32);
    assert_eq!(seg_len, 3);
    assert_eq!(seg_base, 0);

    // Glyph table layout: u16 x, u8 w, i8 advance (as u8 here)
    // g0
    assert_eq!(u16_le(&bytes, glyph_table_offset + 0), 0);
    assert_eq!(bytes[glyph_table_offset + 2], 3);
    assert_eq!(bytes[glyph_table_offset + 3] as i8, 3);
    // g1
    assert_eq!(u16_le(&bytes, glyph_table_offset + 4), 3);
    assert_eq!(bytes[glyph_table_offset + 6], 2);
    assert_eq!(bytes[glyph_table_offset + 7] as i8, 2);
    // g2
    assert_eq!(u16_le(&bytes, glyph_table_offset + 8), 5);
    assert_eq!(bytes[glyph_table_offset + 10], 4);
    assert_eq!(bytes[glyph_table_offset + 11] as i8, 4);

    // Inner atlas header: width, height at atlas_offset
    assert_eq!(u16_le(&bytes, atlas_offset + 0), w);
    assert_eq!(u16_le(&bytes, atlas_offset + 2), h);

    // Kerning block
    assert_eq!(kerning_count, 1);
    assert!(kerning_offset > atlas_offset);
    assert_eq!(total_len, bytes.len());
    assert_eq!(kerning_offset, atlas_offset + atlas_len);

    // Each kerning record: u24 left, u24 right, i8 adj
    let k_left = u24_le(&bytes, kerning_offset + 0);
    let k_right = u24_le(&bytes, kerning_offset + 3);
    let k_adj = bytes[kerning_offset + 6] as i8;
    assert_eq!(k_left, 'A' as u32);
    assert_eq!(k_right, 'B' as u32);
    assert_eq!(k_adj, -1);
  }

  #[test]
  fn two_segments_and_no_kerning() {
    // Two segments: '0'..'1' (2 glyphs), 'x'..'z' (3 glyphs) => 5 total
    let meta = Font {
      line_height: 20,
      ascent: 15,
      descent: -5,
      glyphs: vec![
        // first segment (2)
        Glyph { x: 0, w: 1, advance: None },
        Glyph { x: 1, w: 2, advance: None },
        // second segment (3)
        Glyph { x: 3, w: 3, advance: None },
        Glyph { x: 6, w: 4, advance: None },
        Glyph { x: 10, w: 5, advance: None },
      ],
      kerning: vec![],
      charset: vec![
        CharsetRange { start: "0".into(), end: "1".into() },
        CharsetRange { start: "x".into(), end: "z".into() },
      ],
    };

    let (w, h) = (15u16, 6u16);
    let mut l8 = vec![0u8; (w as usize) * (h as usize)];
    for i in 0..l8.len() {
      l8[i] = (i % 253) as u8;
    }

    let bytes = build_font_from_l8_and_meta(&l8, w, h, &meta).expect("builder ok");

    // Offsets
    let charset_seg_count = u16_le(&bytes, 0x2A) as usize;
    assert_eq!(charset_seg_count, 2);

    let header_len = 0x2C + 7 * charset_seg_count;
    let glyph_table_offset = u32_le(&bytes, 0x0E) as usize;
    let glyph_table_len = u32_le(&bytes, 0x12) as usize;
    let atlas_offset = u32_le(&bytes, 0x16) as usize;
    let atlas_len = u32_le(&bytes, 0x1A) as usize;
    let kerning_offset = u32_le(&bytes, 0x22) as usize;
    let kerning_count = u32_le(&bytes, 0x26) as usize;

    assert_eq!(glyph_table_offset, header_len);
    assert_eq!(glyph_table_len, 5 * 4);
    assert_eq!(atlas_offset, header_len + glyph_table_len);
    assert_eq!(kerning_count, 0);
    assert_eq!(kerning_offset, 0);

    // Verify segments contents and cumulative glyph_base
    let seg0_start = u24_le(&bytes, 0x2C + 0);
    let seg0_len = u16_le(&bytes, 0x2C + 3);
    let seg0_base = u16_le(&bytes, 0x2C + 5);
    assert_eq!(seg0_start, '0' as u32);
    assert_eq!(seg0_len, 2);
    assert_eq!(seg0_base, 0);

    let seg1_off = 0x2C + 7;
    let seg1_start = u24_le(&bytes, seg1_off + 0);
    let seg1_len = u16_le(&bytes, seg1_off + 3);
    let seg1_base = u16_le(&bytes, seg1_off + 5);
    assert_eq!(seg1_start, 'x' as u32);
    assert_eq!(seg1_len, 3);
    assert_eq!(seg1_base, 2); // after first segment length 2

    // Inner atlas dimensions
    assert_eq!(u16_le(&bytes, atlas_offset + 0), w);
    assert_eq!(u16_le(&bytes, atlas_offset + 2), h);

    // Atlas length should be positive and keep file bounds sane
    assert!(atlas_len > 21); // inner header is at least 21 bytes
    assert!(atlas_offset + atlas_len <= bytes.len());
  }

  #[test]
  fn charset_count_mismatch_is_error() {
    // charset claims 3 codepoints, but we provide only 2 glyphs
    let bad = Font {
      line_height: 8,
      ascent: 6,
      descent: -2,
      glyphs: vec![Glyph { x: 0, w: 3, advance: None }, Glyph { x: 3, w: 2, advance: None }],
      kerning: vec![],
      charset: vec![CharsetRange { start: "A".into(), end: "C".into() }],
    };

    let w = 5u16;
    let h = 4u16;
    let l8 = vec![0u8; (w as usize) * (h as usize)];
    let err = build_font_from_l8_and_meta(&l8, w, h, &bad).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("total codepoints in charset") && msg.contains("does not match glyph count"));
  }
}
