use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Parser};
use std::{fs, fs::File, io::Write, path::PathBuf};

use zaf::font::{self, ttfgen};

// ---------------------------------------------
// zaf: Font builder CLI
// Modes:
//   1) --atlas <png> --json <file>               -> build from atlas + JSON
//   2) --ttf <font> --size <px> [--range A:B]*   -> render TTF and build
// Produces: ZFNT container (magic+header+glyphs+inner ZAF atlas + optional kerning)
// ---------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "zaf", author, version, about = "zaf: build ZFNT font (from atlas+JSON or directly from TTF)", long_about=None)]
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

  /// Optional global tracking/letter-spacing in pixels (default: 0)
  #[arg(long = "letter-spacing", default_value_t = 0)]
  letter_spacing: i16,

  /// Input atlas image path (PNG/JPG/etc.) [alternative to --ttf]
  #[arg(short = 'a', long = "atlas")]
  atlas: Option<PathBuf>,

  /// Font config JSON path (glyph positions/advances, metrics) [alternative to --ttf]
  #[arg(short = 'j', long = "json")]
  json: Option<PathBuf>,

  /// Output file (.zfnt)
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
    let (zfnt, png, json_text) =
      ttfgen::build_from_ttf_with_artifacts(&ttf_bytes, size, &ranges, cli.letter_spacing).context("build from TTF")?;

    // Derive sidecar paths from --output (Foo.zfnt -> Foo.atlas.png / Foo.meta.json)
    let base = cli.output.with_extension("");
    let atlas_path = base.with_extension("atlas.png");
    let meta_path = base.with_extension("meta.json");

    fs::write(&atlas_path, &png).with_context(|| format!("write {:?}", atlas_path))?;
    fs::write(&meta_path, &json_text).with_context(|| format!("write {:?}", meta_path))?;
    zfnt
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
    font::build_font_from_png_and_json(&png_bytes, &json_text).context("build ZFNT container")?
  };

  // Write output
  let mut f = File::create(&cli.output).with_context(|| format!("create {:?}", cli.output))?;
  f.write_all(&blob)?;
  f.flush()?;

  eprintln!("wrote {} bytes to {}", blob.len(), cli.output.display());
  Ok(())
}
