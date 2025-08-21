use crate::font::{Atlas, CharsetRange, Glyph, Metadata};
use anyhow::{Result, anyhow};
use rayon::ThreadPoolBuilder;
use std::{collections::HashMap, sync::Arc};
use ttf_parser::Face;
use zng_webrender_api::{
  ColorF, FontInstanceFlags, FontInstanceKey, FontInstanceOptions, FontInstancePlatformOptions, FontKey,
  FontRenderMode, FontTemplate, GlyphIndex, IdNamespace, units::DevicePoint,
};
use zng_wr_glyph_rasterizer::{
  BaseFontInstance, FontInstance, GlyphKey, GlyphRasterizer, RasterizedGlyph, SharedFontResources, SubpixelDirection,
  profiler::GlyphRasterizeProfiler,
};

/// Row in the single-row atlas.
struct Row {
  x: u16,
  top: i32,
  w: i32,
  h: i32,
  bytes: Vec<u8>, // L8 tightly packed (w * h)
}

///  MiniType
pub fn convert_from_ttf(ttf: &[u8], px: f32, ranges: &[(char, char)]) -> Result<(Metadata, Atlas)> {
  if px <= 0.0 {
    return Err(anyhow!("px must be > 0"));
  }

  // 1) Charset
  let charset = expand_charset(ranges)?;

  // 2) Metrics
  let face = Face::parse(ttf, 0)?;
  let upem = face.units_per_em() as f32;

  let (asc_u, desc_u, gap_u) = if let Some(os2) = face.tables().os2 {
    (
      os2.typographic_ascender() as f32,
      os2.typographic_descender() as f32, // usually negative
      os2.typographic_line_gap() as f32,
    )
  } else {
    let hhea = face.tables().hhea;
    (hhea.ascender as f32, hhea.descender as f32, hhea.line_gap as f32)
  };

  let px_per_unit = px / upem;
  let ascent_px = asc_u * px_per_unit;
  let descent_px = (-desc_u) * px_per_unit; // make positive
  let line_gap_px = gap_u.max(0.0) * px_per_unit;

  let line_height: u16 = (ascent_px + descent_px + line_gap_px).ceil().max(1.0) as u16;
  let baseline_y: i32 = ascent_px.ceil() as i32; // y grows down
  let atlas_h: u16 = (ascent_px + descent_px).ceil().max(1.0) as u16;

  // 3) Rasterize (L8)
  let rasters = rasterize_glyphs(ttf, px, &charset)?;
  if rasters.len() != charset.len() {
    return Err(anyhow!("rasterizer returned {} glyphs for {} chars", rasters.len(), charset.len()));
  }

  // 4) Build glyph records + pack single-row atlas
  let mut glyphs_out = Vec::with_capacity(charset.len());
  let mut rows = Vec::with_capacity(charset.len());
  let mut x_cursor: u32 = 0;

  for (i, ch) in charset.iter().enumerate() {
    let rg = &rasters[i];

    let adv_i8 = {
      let adv_px = face
        .glyph_index(*ch)
        .and_then(|gid| face.glyph_hor_advance(gid))
        .map(|u| (u as f32) * px_per_unit)
        .unwrap_or(0.0);
      saturate_i8(adv_px.round())
    };

    if rg.width <= 0 || rg.height <= 0 {
      push_empty_glyph(&mut glyphs_out, x_cursor as u16, adv_i8);
      continue;
    }

    let mut w = rg.width as i32;
    let mut h = rg.height as i32;
    let mut left = rg.left.floor() as i32;
    let mut top = rg.top.floor() as i32;

    // Tighten to non-zero alpha bounds (stride == w; WR L8 is tightly packed)
    if let Some((bytes, cw, ch_crop, dx, dy)) = crop_l8_tight(&rg.bytes, w, h) {
      left += dx; // bytes left columns => bearing increases
      top -= dy; // bytes top rows      => distance from baseline decreases
      w = cw;
      h = ch_crop;

      glyphs_out.push(Glyph::new(
        x_cursor as u16,
        (w as u32).min(255) as u8,
        Some(adv_i8),
        Some(saturate_i8(left as f32)),
      ));

      rows.push(Row { x: x_cursor as u16, top, w, h, bytes });
      x_cursor = x_cursor.saturating_add(w as u32);
    } else {
      // Truly empty after crop
      push_empty_glyph(&mut glyphs_out, x_cursor as u16, adv_i8);
    }
  }

  // 5) Compose L8 atlas
  let atlas_w: u16 = (x_cursor as u16).max(1);
  let mut l8 = vec![0u8; (atlas_w as usize) * (atlas_h as usize)];

  for row in &rows {
    if row.w <= 0 || row.h <= 0 {
      continue;
    }
    let dst_top = baseline_y - row.top;
    let dst_left = row.x as i32;

    for y in 0..row.h {
      let dst_y = dst_top + y;
      if dst_y < 0 || dst_y >= atlas_h as i32 {
        continue;
      }
      let dst_x = dst_left as usize;
      let row_w = row.w as usize;
      let remaining = (atlas_w as usize).saturating_sub(dst_x);
      let copy_w = row_w.min(remaining);
      if copy_w == 0 {
        continue;
      }

      let src_off = (y as usize) * row_w;
      let dst_off = (dst_y as usize) * (atlas_w as usize) + dst_x;

      debug_assert!(src_off + copy_w <= row.bytes.len());
      debug_assert!(dst_off + copy_w <= l8.len());

      l8[dst_off..dst_off + copy_w].copy_from_slice(&row.bytes[src_off..src_off + copy_w]);
    }
  }

  let atlas = Atlas::new(atlas_w, atlas_h, l8)?;

  let charsets: Vec<CharsetRange> = ranges
    .iter()
    .map(|(s, e)| CharsetRange { start: s.to_string(), end: e.to_string() })
    .collect();

  let meta = Metadata {
    line_height,
    ascent: ascent_px.round() as i16,
    descent: descent_px.round() as i16,
    glyphs: glyphs_out,
    kerning: Vec::new(),
    charset: charsets,
  };

  Ok((meta, atlas))
}

fn expand_charset(ranges: &[(char, char)]) -> Result<Vec<char>> {
  let mut out = Vec::new();
  for &(start, end) in ranges {
    if end < start {
      return Err(anyhow!("range end < start: {start:?}-{end:?}"));
    }
    let mut c = start as u32;
    let end_u = end as u32;
    while c <= end_u {
      if let Some(ch) = char::from_u32(c) {
        out.push(ch);
      }
      c += 1;
    }
  }
  Ok(out)
}

/// Rasterize the requested characters from raw font bytes using wr_glyph_rasterizer.
fn rasterize_glyphs(font_bytes: &[u8], px: f32, chars: &[char]) -> Result<Vec<RasterizedGlyph>> {
  if px <= 0.0 {
    return Err(anyhow!("px must be > 0"));
  }

  // Shared font registry
  let namespace = IdNamespace(0);
  let mut fonts = SharedFontResources::new(namespace);

  let font_key = FontKey::new(namespace, 0);
  let font_template = FontTemplate::Raw(Arc::new(font_bytes.to_vec()), 0);
  let shared_font_key = fonts
    .font_keys
    .add_key(&font_key, &font_template)
    .expect("Failed to add font key");
  fonts.templates.add_font(shared_font_key, font_template);

  // Font instance
  let font_instance_key = FontInstanceKey::new(namespace, 1);
  let base = BaseFontInstance::new(
    font_instance_key,
    shared_font_key,
    px,
    Some(FontInstanceOptions::default()),
    Some(FontInstancePlatformOptions::default()),
    Vec::new(), // variations
  );
  let shared_instance = fonts
    .instance_keys
    .add_key(base)
    .expect("Failed to add font instance key");
  fonts.instances.add_font_instance(shared_instance);

  // Workers + rasterizer
  let workers = Arc::new(
    ThreadPoolBuilder::new()
      .thread_name(|i| format!("WRWorker#{i}"))
      .build()
      .map_err(|e| anyhow!("Failed to create worker pool: {e}"))?,
  );

  let mut ras = GlyphRasterizer::new(workers, None, true);
  ras.add_font(
    shared_font_key,
    fonts
      .templates
      .get_font(&shared_font_key)
      .ok_or_else(|| anyhow!("Missing FontTemplate"))?,
  );

  let font = fonts.instances.get_font_instance(font_instance_key).unwrap();
  let mut instance = FontInstance::new(font, ColorF::BLACK.into(), FontRenderMode::Alpha, FontInstanceFlags::empty());
  ras.prepare_font(&mut instance);

  // Map chars → glyph indices → keys (preserve order)
  let keys: Vec<GlyphKey> = chars
    .into_iter()
    .map(|&ch| {
      let gidx: GlyphIndex = ras.get_glyph_index(shared_font_key, ch).unwrap();
      GlyphKey::new(gidx, DevicePoint::zero(), SubpixelDirection::None)
    })
    .collect();

  // Queue + resolve
  ras.request_glyphs(instance, &keys, |_| true);

  let mut map: HashMap<GlyphKey, RasterizedGlyph> = HashMap::with_capacity(keys.len());
  ras.resolve_glyphs(
    |job, _| {
      map.insert(job.key, job.result.unwrap());
    },
    &mut Profiler,
  );

  Ok(keys.into_iter().map(|k| map.remove(&k).unwrap()).collect())
}

struct Profiler;
impl GlyphRasterizeProfiler for Profiler {
  fn start_time(&mut self) {}
  fn end_time(&mut self) -> f64 {
    0.0
  }
  fn set(&mut self, _value: f64) {}
}

/// Crop tightly to non-zero alpha. Returns (bytes, w, h, dx, dy),
/// where (dx, dy) is the offset of the crop rect relative to the original bitmap.
#[inline]
fn crop_l8_tight(bytes: &[u8], w: i32, h: i32) -> Option<(Vec<u8>, i32, i32, i32, i32)> {
  if w <= 0 || h <= 0 {
    return None;
  }
  let (w_us, h_us) = (w as usize, h as usize);

  let mut found_any = false;
  let mut min_x = w_us;
  let mut max_x = 0usize;
  let mut min_y = h_us;
  let mut max_y = 0usize;

  for y in 0..h_us {
    let row = &bytes[y * w_us..(y + 1) * w_us];

    // first/last non-zero in this row
    let mut lx = None;
    let mut rx = None;
    for (x, &p) in row.iter().enumerate() {
      if p != 0 {
        if lx.is_none() {
          lx = Some(x);
        }
        rx = Some(x);
      }
    }
    if let (Some(l), Some(r)) = (lx, rx) {
      found_any = true;
      if y < min_y {
        min_y = y;
      }
      if y > max_y {
        max_y = y;
      }
      if l < min_x {
        min_x = l;
      }
      if r > max_x {
        max_x = r;
      }
    }
  }

  if !found_any {
    return None;
  }

  let cw = (max_x - min_x + 1) as i32;
  let ch = (max_y - min_y + 1) as i32;

  let mut out = vec![0u8; (cw as usize) * (ch as usize)];
  for yy in 0..(ch as usize) {
    let src_off = (min_y + yy) * w_us + min_x;
    let dst_off = yy * (cw as usize);
    out[dst_off..dst_off + (cw as usize)].copy_from_slice(&bytes[src_off..src_off + (cw as usize)]);
  }

  Some((out, cw, ch, min_x as i32, min_y as i32))
}

#[inline]
fn push_empty_glyph(out: &mut Vec<Glyph>, x: u16, adv_i8: i8) {
  out.push(Glyph::new(x, 0, Some(adv_i8), Some(0)));
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
