// mod ct;
mod font;
mod ttfgen;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Parser};
use image::GenericImageView;
use std::{fs, fs::File, io::Write, path::PathBuf};

// ---------------------------------------------
// minitype: Font builder CLI
// Modes:
//   1) --atlas <png> --json <file>               -> build from atlas + JSON
//   2) --ttf <font> --size <px> [--range A:B]*   -> render TTF and build
// Produces: Font container (magic+header+glyphs+inner MiniType atlas + optional kerning)
// ---------------------------------------------
#[derive(Parser, Debug)]
#[command(name = "minitype", author, version, about = "minitype: build MiniType font (from TTF or from atlas+JSON)", long_about=None)]
struct Cli {
  /// Path to a TrueType/OpenType font (alternative to --atlas/--json)
  #[arg(long = "ttf")]
  ttf: Option<PathBuf>,

  /// Pixel size to render the TTF at (required with --ttf)
  #[arg(long = "size")]
  size: Option<f32>,

  /// Character range(s) for TTF mode, inclusive, format START:END (single scalars). Repeatable.
  /// Examples:
  ///   --range " :~"     (printable ASCII incl. space)
  ///   --range "!:~"     (printable ASCII without space)
  ///   --range "0:9" --range "A:Z" --range "a:z"
  #[arg(long = "range", action = ArgAction::Append)]
  ranges: Vec<String>,

  /// Input atlas image path (PNG/JPG/etc.) [alternative to --ttf]
  #[arg(short = 'a', long = "atlas")]
  atlas: Option<PathBuf>,

  /// Font config JSON path (glyph positions/advances, metrics) [alternative to --ttf]
  #[arg(short = 'j', long = "json")]
  json: Option<PathBuf>,

  /// If set, also write preview artifacts next to --output:
  /// <output>.atlas.png and <output>.meta.json (TTF mode only)
  #[arg(long = "preview", default_value_t = false)]
  preview: bool,

  /// Output file (.mtt)
  #[arg(short, long)]
  output: PathBuf,
}

fn parse_ranges(ranges: &[String]) -> Result<Vec<(char, char)>> {
  if ranges.is_empty() {
    // Default: printable ASCII incl. space
    return Ok(vec![(' ', '~')]);
  }
  let mut out = Vec::with_capacity(ranges.len());
  for s in ranges {
    let mut parts = s.split(':');
    let a = parts.next().ok_or_else(|| anyhow!("bad --range: {s}"))?;
    let b = parts.next().ok_or_else(|| anyhow!("bad --range (missing end): {s}"))?;
    if parts.next().is_some() {
      bail!("bad --range (too many colons): {s}");
    }
    let mut ait = a.chars();
    let ac = ait.next().ok_or_else(|| anyhow!("empty start in --range: {s}"))?;
    if ait.next().is_some() {
      bail!("start must be a single scalar: {s}");
    }
    let mut bit = b.chars();
    let bc = bit.next().ok_or_else(|| anyhow!("empty end in --range: {s}"))?;
    if bit.next().is_some() {
      bail!("end must be a single scalar: {s}");
    }
    if bc < ac {
      bail!("range end < start: {s}");
    }
    out.push((ac, bc));
  }
  Ok(out)
}

fn main() -> Result<()> {
  let cli = Cli::parse();

  // Mode selection
  let using_ttf = cli.ttf.is_some();
  let using_atlas = cli.atlas.is_some() || cli.json.is_some();

  if using_ttf && using_atlas {
    bail!("Specify either --ttf/--size/--range... or --atlas/--json, not both");
  }

  let blob = if using_ttf {
    let ttf_path = cli.ttf.as_ref().ok_or_else(|| anyhow!("--ttf is required"))?;
    let size = cli.size.ok_or_else(|| anyhow!("--size is required with --ttf"))?;
    let ranges = parse_ranges(&cli.ranges)?;
    let ttf_bytes = fs::read(ttf_path).with_context(|| format!("read ttf {:?}", ttf_path))?;
    let (meta, atlas) = ttfgen::convert_from_ttf(&ttf_bytes, size, &ranges).context("build from TTF")?;

    // Optionally write preview artifacts when --preview is set
    if cli.preview {
      // Derive sidecar paths from --output (Foo.mtt -> Foo.atlas.png / Foo.meta.json)
      let json_text = serde_json::to_string_pretty(&meta)?;
      let png = encode_l8_png(&atlas.pixels, atlas.width, atlas.height)?;
      let base = cli.output.with_extension("");
      let atlas_path = base.with_extension("atlas.png");
      let meta_path = base.with_extension("meta.json");

      fs::write(&atlas_path, &png).with_context(|| format!("write {:?}", atlas_path))?;
      fs::write(&meta_path, &json_text).with_context(|| format!("write {:?}", meta_path))?;
    }

    font::assemble(meta, atlas)?
  } else {
    let atlas_path = cli
      .atlas
      .as_ref()
      .ok_or_else(|| anyhow!("--atlas is required unless using --ttf"))?;

    let json_path = cli
      .json
      .as_ref()
      .ok_or_else(|| anyhow!("--json is required unless using --ttf"))?;

    let png_bytes = fs::read(atlas_path).with_context(|| format!("read atlas {:?}", atlas_path))?;
    let json_text = fs::read_to_string(json_path).with_context(|| format!("read json {:?}", json_path))?;

    let meta: font::Metadata = serde_json::from_str(&json_text)?;
    let (w, h, l8) = decode_png_to_l8(&png_bytes)?;
    let atlas = font::Atlas::new(w, h, l8)?;

    font::assemble(meta, atlas)?
  };

  // Write output
  let mut f = File::create(&cli.output).with_context(|| format!("create {:?}", cli.output))?;
  f.write_all(&blob)?;
  f.flush()?;

  eprintln!("wrote {} bytes to {}", blob.len(), cli.output.display());
  Ok(())
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

// Encode an L8 grayscale image to PNG bytes.
fn encode_l8_png(pixels: &[u8], w: u16, h: u16) -> Result<Vec<u8>> {
  use image::{ExtendedColorType, ImageEncoder as _, codecs::png::PngEncoder};
  let mut out = Vec::new();
  PngEncoder::new(&mut out).write_image(pixels, w as u32, h as u32, ExtendedColorType::L8)?;
  Ok(out)
}
