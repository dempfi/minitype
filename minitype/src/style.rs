use crate::MiniTypeFont;

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

  /// Returns the vertical offset between the line position and the top edge of the bounding box.
  #[inline]
  fn baseline_offset(&self, baseline: Baseline) -> i32 {
    match baseline {
      Baseline::Top => 0,
      Baseline::Bottom => self.font.atlas_height.saturating_sub(1) as i32,
      Baseline::Middle => (self.font.atlas_height.saturating_sub(1) / 2) as i32,
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
    // Baseline-aligned origin for the *tight* strip.
    let origin = position - Point::new(0, self.baseline_offset(baseline));

    let mut pen_x = origin.x;

    for ch in text.chars() {
      let idx = self
        .font
        .glyph_index_for_cp(ch as u32)
        .or_else(|| self.font.glyph_index_for_cp('?' as u32));

      let Some(idx) = idx else {
        pen_x += 1;
        continue;
      };

      let Some(meta) = self.font.glyph_meta(idx) else {
        pen_x += 1;
        continue;
      };

      // Destination top-left for this glyph strip:
      // - apply left side bearing,
      // - shift down by atlas_y_off because rows are tight-cropped.
      let dst_x0 = pen_x + meta.left as i32;
      let dst_y0 = origin.y + self.font.atlas_y_off as i32;

      if let Some(iter) = self.font.glyph_iter_for_index(idx) {
        let w = iter.width() as usize;
        if w != 0 {
          let mut i = 0usize;
          let mapped = iter.filter_map(|a| {
            let dx = (i % w) as i32;
            let dy = (i / w) as i32;
            i += 1;
            if a == 0 {
              None
            } else {
              Some(Pixel(Point::new(dst_x0 + dx, dst_y0 + dy), Rgba::<C>::new(self.color, a)))
            }
          });
          target.draw_iter(mapped)?;
        }
      }

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
    let position = position - Point::new(0, self.baseline_offset(baseline));
    Ok(position + Point::new(width as i32, self.baseline_offset(baseline)))
  }

  fn measure_string(&self, text: &str, position: Point, baseline: Baseline) -> TextMetrics {
    // Baseline-aligned top of the *tight* band.
    let bb_position = position - Point::new(0, self.baseline_offset(baseline));
    let mut bb_width: u32 = 0;
    let mut min_left: i32 = 0;

    for c in text.chars() {
      let idx = self
        .font
        .glyph_index_for_cp(c as u32)
        .or_else(|| self.font.glyph_index_for_cp('?' as u32));

      if let Some(idx) = idx {
        if let Some(g) = self.font.glyph_meta(idx) {
          bb_width = bb_width.saturating_add(g.advance.max(0) as u32);
          if (g.left as i32) < min_left {
            min_left = g.left as i32;
          }
          continue;
        }
      }
      // Fallback minimal advance if glyph missing
      bb_width = bb_width.saturating_add(1);
    }

    let extra_left = (-min_left).max(0) as u32;

    // Height is the tight band; origin is shifted by atlas_y_off to match draw_string().
    let bb_size = Size::new(bb_width.saturating_add(extra_left), self.font.atlas_height as u32);
    let bb_origin = bb_position + Point::new(min_left, self.font.atlas_y_off as i32);

    TextMetrics {
      bounding_box: Rectangle::new(bb_origin, bb_size),
      next_position: position + Point::new(bb_width as i32, 0),
    }
  }

  fn line_height(&self) -> u32 {
    self.font.line_height as u32
  }
}
