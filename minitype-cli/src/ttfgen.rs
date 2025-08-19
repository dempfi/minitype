// ============================
// TTF → MiniType one-shot builder (artifacts for preview)
// ============================

use crate::font::{Atlas, CharsetRange, Glyph, Metadata, assemble};
use anyhow::anyhow;
use image::ImageEncoder as _;
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Format, Vector};
use swash::{FontRef, NormalizedCoord};

/// Build MiniType (MFNT) plus artifacts: quantized L8 PNG preview and JSON meta.
/// Returns `(mfnt_bytes, png_l8_bytes, json_meta_text)`.
pub fn build_from_ttf_with_artifacts(
  ttf: &[u8],
  px: f32,
  ranges: &[(char, char)],
) -> anyhow::Result<(Vec<u8>, Vec<u8>, String)> {
  // ---- 1) Load font
  let font = FontRef::from_index(ttf, 0).ok_or_else(|| anyhow!("failed to parse TTF font"))?;

  // ---- 2) Expand charset
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

  // ---- 3) Metrics at requested size
  let m_scaled = font.metrics(&[] as &[NormalizedCoord]).scale(px);
  let ascent = m_scaled.ascent;
  let descent = m_scaled.descent;
  let line_gap = m_scaled.leading;
  let line_height = (ascent + descent + line_gap).ceil().max(1.0) as u16;

  // Spacing
  let gmetrics = font.glyph_metrics(&[] as &[NormalizedCoord]).scale(px);

  // ---- 4) Swash render setup (quality over speed)
  let outline = [
    Source::ColorOutline(0),
    Source::ColorBitmap(StrikeWith::BestFit),
    Source::Outline,
  ];
  let mut scale_ctx = ScaleContext::new();
  let mut scaler = scale_ctx
    .builder(font)
    .size(px)
    .hint(true) // rely on Swash hinting for crisp stems
    .build();

  // Toggle: try LCD subpixel and collapse to L8. Safer default is Alpha.
  const TRY_LCD: bool = false;
  let desired_format = if TRY_LCD { Format::Subpixel } else { Format::Alpha };

  // --- helpers for LCD collapse and image scoring ---

  #[inline]
  fn to_linear(x: u8) -> f32 {
    let xf = x as f32 / 255.0;
    if xf <= 0.04045 {
      xf / 12.92
    } else {
      ((xf + 0.055) / 1.055).powf(2.4)
    }
  }
  #[inline]
  fn to_srgb(x: f32) -> u8 {
    let y = if x <= 0.0031308 {
      12.92 * x
    } else {
      1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
  }
  #[inline]
  fn collapse_row_rgb_to_l8(src: &[u8], out: &mut [u8]) {
    // src len == out len * 3
    let mut si = 0;
    for dst in out.iter_mut() {
      let r = to_linear(src[si]);
      let g = to_linear(src[si + 1]);
      let b = to_linear(src[si + 2]);
      si += 3;
      *dst = to_srgb((r + g + b) / 3.0);
    }
  }

  #[inline]
  fn abd(a: u8, b: u8) -> u32 {
    a.max(b) as u32 - a.min(b) as u32
  }

  /// Edge-contrast + midtone penalty: favors crisp, inky glyphs (fewer 30–70% grays).
  fn clarity_score_l8(buf: &[u8], w: i32, h: i32) -> u64 {
    if w <= 1 || h <= 1 {
      return 0;
    }
    let (w, h) = (w as usize, h as usize);
    let mut edge: u64 = 0;
    // horizontal
    for y in 0..h {
      let row = &buf[y * w..y * w + w];
      for x in 1..w {
        edge += abd(row[x], row[x - 1]) as u64;
      }
    }
    // vertical
    for y in 1..h {
      let pr = &buf[(y - 1) * w..(y - 1) * w + w];
      let rw = &buf[y * w..y * w + w];
      for x in 0..w {
        edge += abd(rw[x], pr[x]) as u64;
      }
    }
    // mid-tone penalty
    let mut mid: u64 = 0;
    for &a in buf.iter().take(w * h) {
      if !(a <= 24 || a >= 232) {
        mid += u8::min(a, 255 - a) as u64;
      }
    }
    edge.saturating_mul(3) - mid.saturating_mul(4)
  }

  /// Render one candidate at (ox,oy), producing L8 data (converting from LCD if needed) and a score.
  fn render_candidate(
    scaler: &mut swash::scale::Scaler,
    outline: &[Source],
    gid: u16,
    format: Format,
    ox: f32,
    oy: f32,
  ) -> Option<(i32, i32, i32, i32, Vec<u8>, u64)> {
    let mut r = Render::new(outline);
    r.format(format).offset(Vector::new(ox, oy));
    let img = r.render(scaler, gid)?;
    let w = img.placement.width.max(0) as i32;
    let h = img.placement.height.max(0) as i32;
    let left = img.placement.left as i32;
    let top = img.placement.top as i32;
    if w <= 0 || h <= 0 {
      return Some((w, h, left, top, Vec::new(), 0));
    }

    let mut l8 = if format == Format::Subpixel {
      let (wu, hu) = (w as usize, h as usize);
      let need = wu.saturating_mul(hu).saturating_mul(3);
      if need != img.data.len() {
        // fall back to grayscale if the buffer isn’t exactly WXH*3
        let mut r2 = Render::new(outline);
        r2.format(Format::Alpha).offset(Vector::new(ox, oy));
        if let Some(img2) = r2.render(scaler, gid) {
          img2.data.clone()
        } else {
          Vec::new()
        }
      } else {
        let mut out = vec![0u8; wu * hu];
        for row in 0..hu {
          let src = &img.data[row * (wu * 3)..row * (wu * 3) + (wu * 3)];
          let dst = &mut out[row * wu..row * wu + wu];
          collapse_row_rgb_to_l8(src, dst);
        }
        out
      }
    } else {
      img.data.clone()
    };

    // gentle S-curve: lift black point a hair, nudge mids darker
    soften_haze_curve_in_place(&mut l8, /*cut=*/ 4, /*gamma=*/ 1.1);

    let score = clarity_score_l8(&l8, w, h);
    Some((w, h, left, top, l8, score))
  }

  // Baseline snap suggestion:
  let oy_base: f32 = (ascent.round() - ascent) as f32;
  // Safety clamp (stay well within half a pixel)
  const OY_CLAMP: f32 = 0.45;
  let oy_center = oy_base.clamp(-OY_CLAMP, OY_CLAMP);

  // X phases are reused for scoring at each candidate oy
  const OX_CANDIDATES: &[f32] = &[0.0, -0.375, -0.25, -0.125, 0.125, 0.25, 0.375];

  // Score a given oy by sampling up to N glyphs and taking the best X phase per glyph.
  let sample_count = core::cmp::min(charset.len(), 96);
  let mut score_oy = |oy: f32| -> u128 {
    let mut total: u128 = 0;
    for i in 0..sample_count {
      let gid = font.charmap().map(charset[i]);
      let mut best_for_glyph: u64 = 0;
      for &ox in OX_CANDIDATES {
        if let Some((_w, _h, _l, _t, _buf, sc)) = render_candidate(&mut scaler, &outline, gid, desired_format, ox, oy) {
          if sc > best_for_glyph {
            best_for_glyph = sc;
          }
        }
      }
      total += best_for_glyph as u128;
    }
    total
  };

  // --- Coarse sweep around oy_center
  const OY_COARSE_STEP: f32 = 0.125; // ~1/8 px
  const OY_COARSE_RADIUS_STEPS: i32 = 3; // sweep center ± 3 steps
  let mut best_oy = 0.0f32;
  let mut best_score: u128 = 0;

  for k in -OY_COARSE_RADIUS_STEPS..=OY_COARSE_RADIUS_STEPS {
    let oy = (oy_center + k as f32 * OY_COARSE_STEP).clamp(-OY_CLAMP, OY_CLAMP);
    let s = score_oy(oy);
    if s > best_score || (k == -OY_COARSE_RADIUS_STEPS && best_score == 0) {
      best_score = s;
      best_oy = oy;
    }
  }

  // --- Refine sweep around the coarse winner
  const OY_REFINE_STEP: f32 = 0.03125; // ~1/32 px
  const OY_REFINE_RADIUS_STEPS: i32 = 4; // ± 4 * refine step ≈ ±0.125 px window
  for k in -OY_REFINE_RADIUS_STEPS..=OY_REFINE_RADIUS_STEPS {
    let oy = (best_oy + k as f32 * OY_REFINE_STEP).clamp(-OY_CLAMP, OY_CLAMP);
    let s = score_oy(oy);
    if s > best_score {
      best_score = s;
      best_oy = oy;
    }
  }

  // --- Dead-zone: if result is tiny, prefer 0.0 (avoids imperceptible drift)
  const OY_DEADZONE: f32 = 0.02;
  let oy_global = if best_oy.abs() < OY_DEADZONE { 0.0 } else { best_oy };

  let mut glyphs_vec: Vec<Glyph> = Vec::with_capacity(charset.len());
  let mut x_cursor: u32 = 0;
  let mut bitmaps: Vec<(u16, i32, i32, i32, i32, Vec<u8>)> = Vec::with_capacity(charset.len());

  for &ch in charset.iter() {
    let gid = font.charmap().map(ch);

    // Try a few X subpixel phases with the chosen global Y offset; keep the crispiest.
    let mut best: Option<(i32, i32, i32, i32, Vec<u8>, u64)> = None;
    for &ox in OX_CANDIDATES {
      if let Some(cand) = render_candidate(&mut scaler, &outline, gid, desired_format, ox, oy_global) {
        best = match best {
          None => Some(cand),
          Some(prev) => {
            if cand.5 > prev.5 {
              Some(cand)
            } else {
              Some(prev)
            }
          }
        };
      }
    }

    if let Some((w, h, left, top, gb, _score)) = best {
      // Advance width at requested size
      let adv_px = gmetrics.advance_width(gid);
      glyphs_vec.push(Glyph::new(
        x_cursor as u16,
        w.clamp(0, 255) as u8,
        Some(saturate_i8(adv_px)),
        Some(saturate_i8(left as f32)),
      ));

      bitmaps.push((x_cursor as u16, left, top, w, h, gb));
      x_cursor = x_cursor.saturating_add(w.max(0) as u32);
    } else {
      // Nothing rendered (e.g., missing glyph)
      glyphs_vec.push(Glyph::new(x_cursor as u16, 0, Some(0), Some(0)));
    }
  }

  let atlas_w: u16 = (x_cursor as u16).max(1);
  let baseline_y = ascent.ceil() as i32;
  let atlas_h = (ascent + descent).ceil().max(1.0) as u16;

  // ---- 6) Compose single-row L8 atlas
  let mut l8 = vec![0u8; atlas_w as usize * atlas_h as usize];
  for (x, _left, top, w, h, bm) in &bitmaps {
    if *w <= 0 || *h <= 0 {
      continue;
    }
    let dst_top = baseline_y - *top;
    let dst_left = *x as i32;
    for row in 0..*h {
      let dst_y = dst_top + row;
      if !(0..(atlas_h as i32)).contains(&dst_y) {
        continue;
      }
      let dst_x = dst_left as usize;
      let row_w = *w as usize;
      let remaining = (atlas_w as usize).saturating_sub(dst_x);
      let copy_w = core::cmp::min(row_w, remaining);
      if copy_w > 0 {
        let src_off = (row as usize) * row_w;
        let dst_off = (dst_y as usize) * (atlas_w as usize) + dst_x;
        let src = &bm[src_off..src_off + copy_w];
        let dst = &mut l8[dst_off..dst_off + copy_w];
        dst.copy_from_slice(src);
      }
    }
  }

  // ---- 7) JSON meta
  let charset_json: Vec<CharsetRange> = ranges
    .iter()
    .map(|(s, e)| CharsetRange { start: s.to_string(), end: e.to_string() })
    .collect();
  let meta = Metadata {
    line_height,
    ascent: ascent.round() as i16,
    descent: descent.round() as i16,
    glyphs: glyphs_vec,
    kerning: Vec::new(),
    charset: charset_json,
  };
  let json_text = serde_json::to_string_pretty(&meta)?;

  // ---- 8) PNG preview (L8)
  let png = encode_l8_png(&l8, atlas_w, atlas_h)?;

  // ---- 9) Build MFNT from L8 + meta
  let atlas = Atlas::new(atlas_w, atlas_h, l8)?;
  let mfnt = assemble(meta, atlas)?;

  Ok((mfnt, png, json_text))
}

/// Encode an L8 grayscale image to PNG bytes.
fn encode_l8_png(pixels: &[u8], w: u16, h: u16) -> anyhow::Result<Vec<u8>> {
  use image::{ExtendedColorType, codecs::png::PngEncoder};
  let mut out = Vec::new();
  PngEncoder::new(&mut out).write_image(pixels, w as u32, h as u32, ExtendedColorType::L8)?;
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

/// Apply a gentle S-curve to reduce haze.
/// `cut` lifts the black point a little (0..30 recommended).
/// `gamma` (>1) darkens mid-tones slightly.
fn soften_haze_curve_in_place(buf: &mut [u8], cut: u8, gamma: f32) {
  let inv = 1.0 / 255.0;
  let cutf = cut as f32 * inv;
  for p in buf.iter_mut() {
    let x = *p as f32 * inv;
    // piecewise: below cut → 0; above cut → re-normalize to 0..1 and apply gamma
    let y = if x <= cutf {
      0.0
    } else {
      ((x - cutf) / (1.0 - cutf)).powf(gamma)
    };
    *p = (y * 255.0 + 0.5).floor() as u8;
  }
}
