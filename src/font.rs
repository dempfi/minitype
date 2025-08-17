/// Font container builder and serializer for ZAF atlas.
///
/// This module defines a compact, MCU‑friendly **font container** that bundles:
/// - the compressed **glyph atlas** (inner ZAF atlas produced by `compress_l8`),
/// - **glyph metrics & mapping** (per‑glyph rectangles, layout metrics, and advance),
/// - global **font metrics** (line height, ascent, descent, letter spacing),
/// - optional **kerning pairs** (left/right glyph codepoints and adjustment),
/// - **charset segments** to map Unicode ranges to glyph indices,
///
/// # Binary container layout (little‑endian)
/// ```text
/// // Offsets are from the start of the container
/// 0x00 4  MAGIC = "ZFNT"
/// 0x04 1  VERSION = 1
/// 0x05 1  FLAGS (Reserved = 0)
/// 0x06 2  line_height (u16)
/// 0x08 2  letter_spacing (i16)
/// 0x0A 2  ascent (i16)
/// 0x0C 2  descent (i16)
/// 0x0E 2  glyph_count (u16)
/// 0x10 4  glyph_table_offset (u32)
/// 0x14 4  glyph_table_len    (u32)   // bytes; each record is 4 bytes: u16 x, u8 w, i8 advance
/// 0x18 4  atlas_offset       (u32)
/// 0x1C 4  atlas_len          (u32)   // bytes; see inner ZAF atlas header below
/// 0x20 4  total_len          (u32)
/// 0x24 4  kerning_offset     (u32)   // 0 if none
/// 0x28 4  kerning_count      (u32)   // number of pairs; each is 7 bytes (u24 left, u24 right, i8 adj)
/// 0x2C 2  charset_seg_count  (u16)
/// 0x2E .. charset_segments[charset_seg_count]  // each 7 bytes: u24 start, u16 len, u16 base
/// ...   .. glyph_table                        // 4 bytes per glyph: u16 x, u8 w, i8 advance
/// ...   .. atlas_blob
/// ...   .. kerning_pairs[kerning_count]       // optional, after atlas blob
/// ```
///
/// // Inner ZAF atlas header (at atlas_offset):
/// // 0  ..1   width  (u16, LE)
/// // 2  ..3   height (u16, LE)
/// // 4        k      (u8)    Global Rice parameter, 0..7
/// // 5  ..20  palette[16] bytes (u8 grayscale values)
/// // 21 ..    payload bits (LSB-first bitstream)
use image::GenericImageView;

use crate::atlas::compress_l8;
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
///   "letter_spacing": 1,
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
  pub letter_spacing: i16,
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
  fn resolved_advance(&self) -> i8 {
    self.advance.unwrap_or_else(|| self.w.min(u8::MAX) as i8)
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

/// Core builder: takes L8 coverage pixels and Font meta, compresses the atlas,
/// and serializes the ZFNT container.
pub fn build_font_from_l8_and_meta(l8: &[u8], width: u16, height: u16, meta: &Font) -> anyhow::Result<Vec<u8>> {
  // Reuse the inner ZAF atlas compressor.
  let atlas_blob = compress_l8(l8, width, height);

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

  // Fixed header is 0x2E bytes, then charset segments (7 bytes each)
  let fixed_header_len: usize = 0x2E; // 46 bytes
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
  out.extend_from_slice(b"ZFNT"); // MAGIC
  out.push(1); // VERSION
  out.push(0); // FLAGS
  out.extend_from_slice(&meta.line_height.to_le_bytes());
  out.extend_from_slice(&meta.letter_spacing.to_le_bytes());
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

  // total_len at 0x20
  let total_len = out.len() as u32;
  out[0x20..0x24].copy_from_slice(&total_len.to_le_bytes());

  // kerning_offset at 0x24
  out[0x24..0x28].copy_from_slice(&kerning_offset.to_le_bytes());

  // kerning_count at 0x28
  out[0x28..0x2C].copy_from_slice(&kerning_count.to_le_bytes());

  Ok(out)
}

/// Build a ZFNT font container from an atlas PNG and a JSON metadata string.
/// Returns the serialized container bytes.
pub fn build_font_from_png_and_json(png_bytes: &[u8], json: &str) -> anyhow::Result<Vec<u8>> {
  let meta: Font = serde_json::from_str(json)?;
  let (w, h, l8) = decode_png_to_l8(png_bytes)?;
  build_font_from_l8_and_meta(&l8, w, h, &meta)
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

/// Decode a PNG (or any supported format) into 8‑bit grayscale, using alpha as
/// coverage when present. Implemented via the `image` crate.
fn decode_png_to_l8(bytes: &[u8]) -> anyhow::Result<(u16, u16, Vec<u8>)> {
  use image::DynamicImage;
  use image::ImageReader;
  use std::io::Cursor;

  let img = ImageReader::new(Cursor::new(bytes))
    .with_guessed_format()? // uses magic header to detect PNG/JPEG/etc.
    .decode()?; // -> DynamicImage

  let (w, h) = img.dimensions();
  let w16: u16 = w.try_into().map_err(|_| anyhow::anyhow!("atlas width too large"))?;
  let h16: u16 = h.try_into().map_err(|_| anyhow::anyhow!("atlas height too large"))?;

  let l8: Vec<u8> = match img {
    DynamicImage::ImageLuma8(l) => l.into_vec(),
    DynamicImage::ImageLumaA8(la) => {
      // Use alpha as coverage.
      let data = la.into_vec();
      let mut v = Vec::with_capacity(data.len() / 2);
      for chunk in data.chunks_exact(2) {
        v.push(chunk[1]);
      }
      v
    }
    DynamicImage::ImageRgba8(rgba) => {
      // Use alpha as coverage.
      let data = rgba.into_vec();
      let mut v = Vec::with_capacity(data.len() / 4);
      for chunk in data.chunks_exact(4) {
        v.push(chunk[3]);
      }
      v
    }
    DynamicImage::ImageRgb8(rgb) => {
      // Convert RGB to luma (BT.709) and clamp.
      let data = rgb.into_vec();
      let mut v = Vec::with_capacity(data.len() / 3);
      for chunk in data.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        v.push(y.round().clamp(0.0, 255.0) as u8);
      }
      v
    }
    other => {
      // Fallback: convert to RGBA8 and take alpha as coverage.
      let rgba = other.to_rgba8();
      let data = rgba.into_vec();
      let mut v = Vec::with_capacity(data.len() / 4);
      for chunk in data.chunks_exact(4) {
        v.push(chunk[3]);
      }
      v
    }
  };

  Ok((w16, h16, l8))
}

/// ---
/// TTF -> ZFNT one-shot builder
///
/// This module renders a TrueType/OpenType font at a fixed pixel size into a
/// single-row grayscale atlas, collects glyph advances and optional kerning,
/// and then invokes `build_font_from_png_and_json` to produce the final ZFNT
/// container bytes.
pub mod ttfgen {
  use super::{CharsetRange, Font, Glyph, Kerning};
  use anyhow::anyhow;
  use image::ImageEncoder;

  /// Build a ZFNT container and sidecar artifacts (PNG atlas + JSON meta)
  /// directly from a TTF/OTF bytestream.
  /// Returns (zfnt_bytes, png_atlas_bytes, json_meta_text).
  pub fn build_from_ttf_with_artifacts(
    ttf: &[u8],
    px: f32,
    ranges: &[(char, char)],
    letter_spacing: i16,
  ) -> anyhow::Result<(Vec<u8>, Vec<u8>, String)> {
    // 1) Load font
    let settings = fontdue::FontSettings { scale: px, ..Default::default() };
    let font = fontdue::Font::from_bytes(ttf, settings).map_err(|_| anyhow!("failed to parse TTF font"))?;

    // 2) Collect charset in order
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

    // 3) Font vertical metrics
    let lm = font
      .horizontal_line_metrics(px)
      .ok_or_else(|| anyhow!("font has no horizontal line metrics"))?;
    let ascent = lm.ascent;
    let descent = lm.descent;
    let line_gap = lm.line_gap;
    let line_height = (ascent - descent + line_gap).ceil().max(1.0) as u16;

    // Place the baseline `ceil(ascent)` pixels from the top of the atlas.
    // Atlas height covers full ascent + |descent| using ceil to avoid clipping.
    let baseline_y = ascent.ceil() as i32;
    let atlas_h = (ascent - descent).ceil().max(1.0) as u16;

    // Spacing constant for inter-glyph gap in atlas
    const SPACING_PX: u32 = 1; // inter-glyph spacing in the generated PNG/atlas

    // 4) Rasterize glyphs and accumulate widths, with spacing between glyphs
    let mut glyphs_vec: Vec<Glyph> = Vec::with_capacity(charset.len());
    let mut x_cursor: u32 = 0;
    let mut bitmaps: Vec<(u16, fontdue::Metrics, Vec<u8>)> = Vec::with_capacity(charset.len());
    for (i, ch) in charset.iter().enumerate() {
      let (metrics, bitmap) = font.rasterize(*ch, px);
      let w = metrics.width as u8;
      glyphs_vec.push(Glyph {
        x: x_cursor as u16,
        w,
        advance: Some(super::ttfgen::saturate_i8(metrics.advance_width)),
      });
      bitmaps.push((x_cursor as u16, metrics, bitmap));
      // advance cursor by glyph width
      x_cursor = x_cursor.saturating_add(metrics.width as u32);
      // add spacing between glyphs, but not after the last one
      if i + 1 < charset.len() {
        x_cursor = x_cursor.saturating_add(SPACING_PX);
      }
    }
    let atlas_w: u16 = (x_cursor as u16).max(1);

    // 5) Compose L8 atlas
    let mut l8 = vec![0u8; atlas_w as usize * atlas_h as usize];
    for (x, m, bm) in &bitmaps {
      if m.width == 0 || m.height == 0 {
        continue;
      }
      // fontdue: `ymin` is the offset of the bottom-most bitmap row from the baseline.
      // Top-of-bitmap relative to the atlas is: baseline_y - (height + ymin).
      let top = baseline_y - (m.height as i32 + m.ymin as i32);
      let left = *x as i32;
      for row in 0..m.height as i32 {
        let dst_y = top + row;
        if dst_y < 0 || dst_y >= atlas_h as i32 {
          continue;
        }
        let src_off = (row as usize) * (m.width as usize);
        let dst_off = (dst_y as usize) * (atlas_w as usize) + (left as usize);
        l8[dst_off..dst_off + (m.width as usize)].copy_from_slice(&bm[src_off..src_off + (m.width as usize)]);
      }
    }

    // 6) Build JSON meta
    let charset_json: Vec<CharsetRange> = ranges
      .iter()
      .map(|(s, e)| CharsetRange { start: s.to_string(), end: e.to_string() })
      .collect();
    let meta = Font {
      line_height,
      letter_spacing,
      ascent: ascent.round() as i16,
      descent: descent.round() as i16,
      glyphs: glyphs_vec,
      kerning: compute_kerning_pairs(&font, px, &charset)?,
      charset: charset_json,
    };
    let json_text = serde_json::to_string_pretty(&meta)?;

    // 7) Encode PNG atlas (with 2px gaps between glyphs to prevent sampling bleed)
    let png_bytes = encode_l8_png(&l8, atlas_w, atlas_h)?;

    // 8) Build ZFNT bytes via the core path
    let zfnt = super::build_font_from_l8_and_meta(&l8, atlas_w, atlas_h, &meta)?;
    Ok((zfnt, png_bytes, json_text))
  }

  /// Encode an L8 grayscale image to PNG bytes.
  fn encode_l8_png(pixels: &[u8], w: u16, h: u16) -> anyhow::Result<Vec<u8>> {
    use image::{ExtendedColorType, codecs::png::PngEncoder};
    let mut out = Vec::new();
    let enc = PngEncoder::new(&mut out);
    enc.write_image(pixels, w as u32, h as u32, ExtendedColorType::L8)?;
    Ok(out)
  }

  /// Build a ZFNT container directly from a TTF/OTF bytestream.
  ///
  /// * `ttf`  – TrueType/OpenType font bytes
  /// * `px`   – target pixel size (height) for rasterization
  /// * `ranges` – character ranges (inclusive), e.g. [(' ', '~')]
  /// * `letter_spacing` – global tracking in pixels
  pub fn build_from_ttf(ttf: &[u8], px: f32, ranges: &[(char, char)], letter_spacing: i16) -> anyhow::Result<Vec<u8>> {
    let (zfnt, _png, _json) = build_from_ttf_with_artifacts(ttf, px, ranges, letter_spacing)?;
    Ok(zfnt)
  }

  #[inline]
  fn saturate_i8(v: f32) -> i8 {
    let r = v.round();
    if r < i8::MIN as f32 {
      i8::MIN
    } else if r > i8::MAX as f32 {
      i8::MAX
    } else {
      r as i8
    }
  }

  /// Compute kerning pairs for the provided charset.
  /// NOTE: To keep size in check, we only include pairs with |adj| >= 0.5 px.
  fn compute_kerning_pairs(font: &fontdue::Font, px: f32, charset: &[char]) -> anyhow::Result<Vec<Kerning>> {
    let mut out = Vec::new();
    for &a in charset {
      for &b in charset {
        let k = font.horizontal_kern(a, b, px).unwrap_or(0.0); // fontdue provides kerning; returns f32
        if k.abs() >= 0.5 {
          out.push(Kerning { left: a.to_string(), right: b.to_string(), adj: saturate_i8(k) });
        }
      }
    }
    Ok(out)
  }
}
