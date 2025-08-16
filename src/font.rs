/// Font container builder and serializer for ZAF atlas.
///
/// This module defines a compact, MCU‑friendly **font container** that bundles:
/// - the compressed **glyph atlas** (inner ZAF atlas produced by `compress_l8`), and
/// - **glyph metrics & mapping** (per‑glyph rectangles and layout metrics), and
/// - global **font metrics** (line height, ascent, descent, letter spacing).
///
/// # Binary container layout (little‑endian)
/// ```text
/// // Offsets are from the start of the container
/// 0x00  4   MAGIC = b"ZFNT"
/// 0x04  1   VERSION = 1
/// 0x05  1   FLAGS (reserved, 0)
/// 0x06  2   line_height (u16)
/// 0x08  2   letter_spacing (i16)
/// 0x0A  2   ascent (i16)
/// 0x0C  2   descent (i16)
/// 0x0E  2   glyph_count (u16)
/// 0x10  4   glyph_table_offset (u32)
/// 0x14  4   atlas_offset (u32)         // start of inner ZAF atlas blob (21+ bytes)
/// 0x18  4   total_len (u32)            // for quick bounds checks
/// 0x1C  ..  glyph_table[glyph_count]   // packed records, 6 bytes each (see below)
/// ...  ..  atlas_blob                  // inner ZAF bytestream from `compress_l8`
///
/// // Glyph record (6 bytes each) — baseline baked; atlas height = full ascent+descent:
/// 0   3   codepoint (u24 LE)           // Unicode scalar value (0..=0x10_FFFF)
/// 3   2   x (u16)
/// 5   1   w (u8)
/// ```
///
/// **Note:** The atlas is a single-row strip with the baseline baked in; renderers can place
/// glyphs at `y=0` and use the atlas height for the full ascent+descent. Vertical offsets are
/// therefore unnecessary.
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
///   "glyphs": [
///     {"ch":"A", "x":0, "w":12}
///   ]
/// }
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FontJson {
  pub line_height: u16,
  pub letter_spacing: i16,
  pub ascent: i16,
  pub descent: i16,
  pub glyphs: Vec<JsonGlyph>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct JsonGlyph {
  /// Single Unicode character for this glyph (must be exactly one scalar).
  pub ch: String,
  pub x: u16,
  pub w: u8,
}

impl JsonGlyph {
  fn resolved_codepoint(&self) -> anyhow::Result<u32> {
    let mut it = self.ch.chars();
    let c = it.next().ok_or_else(|| anyhow::anyhow!("glyph 'ch' is empty"))?;
    if it.next().is_some() {
      return Err(anyhow::anyhow!("glyph 'ch' must be exactly one Unicode scalar"));
    }
    Ok(c as u32)
  }
}

/// Build a ZFNT font container from an atlas PNG and a JSON metadata string.
/// Returns the serialized container bytes.
pub fn build_font_from_png_and_json(png_bytes: &[u8], json: &str) -> anyhow::Result<Vec<u8>> {
  let meta: FontJson = serde_json::from_str(json)?;
  let (w, h, l8) = decode_png_to_l8(png_bytes)?;

  // Reuse the inner ZAF atlas compressor.
  let atlas_blob = compress_l8(&l8, w, h);

  // Prepare glyph table (6 bytes each).
  let glyph_count: u16 = meta
    .glyphs
    .len()
    .try_into()
    .map_err(|_| anyhow::anyhow!("too many glyphs"))?;

  let header_len: usize = 0x1C; // 28 bytes before glyph table
  let glyph_rec_len: usize = 6; // 6 bytes per glyph record (u24 codepoint)
  let glyph_table_len = glyph_rec_len * (glyph_count as usize);

  let glyph_table_offset: u32 = header_len as u32;
  let atlas_offset: u32 = (header_len + glyph_table_len) as u32;

  let mut out = Vec::with_capacity(header_len + glyph_table_len + atlas_blob.len());

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
  out.extend_from_slice(&atlas_offset.to_le_bytes());
  out.extend_from_slice(&(0u32).to_le_bytes()); // total_len placeholder

  // Glyph table
  for g in &meta.glyphs {
    let cp = g.resolved_codepoint()?;
    push_u24_le(&mut out, cp); // codepoint u24 LE
    out.extend_from_slice(&g.x.to_le_bytes()); // x
    out.push(g.w); // w (u8)
  }

  // Atlas blob
  out.extend_from_slice(&atlas_blob);

  // total_len
  let total_len = out.len() as u32;
  let len_bytes = total_len.to_le_bytes();
  out[0x18..0x1C].copy_from_slice(&len_bytes);

  Ok(out)
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
