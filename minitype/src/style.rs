use crate::{GlyphId, MiniTypeFont};

use embedded_graphics::text::{Baseline, renderer::*};
use embedded_graphics_core::{prelude::*, primitives::*};
use embedded_rgba::{Blend, Rgba};

#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MiniTextStyle<'a, C>
where
  C: RgbColor,
  Rgba<C>: Blend<C>,
{
  pub color: C,
  pub font: &'a MiniTypeFont<'a>,
}

impl<'a, C> MiniTextStyle<'a, C>
where
  C: RgbColor,
  Rgba<C>: Blend<C>,
{
  /// Creates a text style with transparent background.
  pub const fn new(font: &'a MiniTypeFont<'a>, color: C) -> Self {
    Self { color, font }
  }

  /// Vertical offset (in px) from the provided baseline position to the *top of the line box*.
  #[inline]
  fn baseline_offset(&self, baseline: Baseline) -> i32 {
    match baseline {
      Baseline::Top => 0,
      Baseline::Bottom => self.font.line_height.saturating_sub(1) as i32,
      Baseline::Middle => (self.font.line_height.saturating_sub(1) / 2) as i32,
      Baseline::Alphabetic => self.font.ascent as i32,
    }
  }
}

impl<C> TextRenderer for MiniTextStyle<'_, C>
where
  C: RgbColor,
  Rgba<C>: Blend<C>,
{
  type Color = Rgba<C>;

  fn draw_string<D>(&self, text: &str, position: Point, baseline: Baseline, target: &mut D) -> Result<Point, D::Error>
  where
    D: DrawTarget<Color = Self::Color>,
  {
    // `origin` is the *top of the line box* that corresponds to `position`/`baseline`.
    let origin = position - Point::new(0, self.baseline_offset(baseline));

    let mut pen_x = origin.x;

    for ch in text.chars() {
      let glyph_index = self
        .font
        .glyph(ch)
        .or_else(|| self.font.glyph('?'))
        .unwrap_or(GlyphId(0));

      let Some(meta) = self.font.metrics(glyph_index) else {
        pen_x += 1;
        continue;
      };

      if let Some(iter) = self.font.glyph_pixels(glyph_index) {
        // Horizontal placement: apply left side bearing (i4).
        let dst_x0 = pen_x + meta.left as i32;

        // Vertical placement: yoff is from line top to first tight row.
        let dst_y0 = origin.y + meta.yoff as i32;

        let w = meta.w as usize;
        if w != 0 {
          let mut i = 0usize;
          let mapped = iter.filter_map(|a| {
            let dx = (i % w) as i32;
            let dy = (i / w) as i32;
            i += 1;
            if a == 0 {
              None
            } else {
              Some(Pixel(Point::new(dst_x0 + dx, dst_y0 + dy), Rgba::new(self.color, a)))
            }
          });
          target.draw_iter(mapped)?;
        }
      }

      // Advance is derived (w + i5 delta) and stored as i16.
      pen_x += meta.advance as i32;
    }

    Ok(Point::new(pen_x, position.y))
  }

  fn draw_whitespace<D>(
    &self,
    width: u32,
    position: Point,
    baseline: Baseline,
    _target: &mut D,
  ) -> Result<Point, D::Error>
  where
    D: DrawTarget<Color = Self::Color>,
  {
    // Keep baseline math consistent with draw_string (no rendering).
    let _ = baseline;
    Ok(Point::new(position.x + width as i32, position.y))
  }

  fn measure_string(&self, text: &str, position: Point, baseline: Baseline) -> TextMetrics {
    // Top of line box aligned to the requested baseline.
    let line_top = position - Point::new(0, self.baseline_offset(baseline));

    let mut bb_width: u32 = 0;
    let mut min_left: i32 = 0;

    // Track vertical extent relative to line_top using new semantics:
    // top_from_line = yoff; bottom_from_line = yoff + h
    let mut min_top_from_line: i32 = i32::MAX;
    let mut max_bottom_from_line: i32 = 0;

    for c in text.chars() {
      let glyph_index = self
        .font
        .glyph(c)
        .or_else(|| self.font.glyph('?'))
        .unwrap_or(GlyphId(0));

      if let Some(g) = self.font.metrics(glyph_index) {
        // accumulate width using signed advance (clamp at 0 for width sum)
        bb_width = bb_width.saturating_add(g.advance.max(0) as u32);

        if (g.left as i32) < min_left {
          min_left = g.left as i32;
        }

        let top_from_line = g.yoff as i32;
        let bottom_from_line = top_from_line + g.h as i32;

        if top_from_line < min_top_from_line {
          min_top_from_line = top_from_line;
        }
        if bottom_from_line > max_bottom_from_line {
          max_bottom_from_line = bottom_from_line;
        }

        continue;
      }

      // Fallback minimal advance if glyph missing
      bb_width = bb_width.saturating_add(1);
    }

    // Handle empty strings (no glyphs produced): height 0 box at line_top.
    if min_top_from_line == i32::MAX {
      return TextMetrics {
        bounding_box: Rectangle::new(line_top, Size::new(0, 0)),
        next_position: position + Point::new(bb_width as i32, 0),
      };
    }

    let extra_left = (-min_left).max(0) as u32;

    let bb_height = (max_bottom_from_line - min_top_from_line).max(0) as u32;
    let bb_size = Size::new(bb_width.saturating_add(extra_left), bb_height);
    let bb_origin = line_top + Point::new(min_left, min_top_from_line);

    TextMetrics {
      bounding_box: Rectangle::new(bb_origin, bb_size),
      next_position: position + Point::new(bb_width as i32, 0),
    }
  }

  fn line_height(&self) -> u32 {
    self.font.line_height as u32
  }
}
