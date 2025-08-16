use anyhow::{Context, Result};
use clap::Parser;
use std::{fs, fs::File, io::Write, path::PathBuf};

use zaf::font;

// ---------------------------------------------
// zaf: Font builder CLI
// - Reads: atlas image (PNG/JPG/etc.) + JSON font config
// - Produces: ZFNT container (magic+header+glyph table+inner ZAF atlas)
// ---------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "zaf", author, version, about = "zaf: build ZAF font from atlas + JSON", long_about=None)]
struct Cli {
  /// Input atlas image path (PNG/JPG/etc.)
  #[arg(short = 'a', long = "atlas")]
  atlas: PathBuf,
  /// Font config JSON path (glyph positions/advances, metrics)
  #[arg(short = 'j', long = "json")]
  json: PathBuf,
  /// Output file (.zaf)
  #[arg(short, long)]
  output: PathBuf,
}

fn main() -> Result<()> {
  let cli = Cli::parse();

  // Read inputs
  let png_bytes = fs::read(&cli.atlas).with_context(|| format!("read atlas {:?}", cli.atlas))?;
  let json_text = fs::read_to_string(&cli.json).with_context(|| format!("read json {:?}", cli.json))?;

  // Build ZAF container
  let blob = font::build_font_from_png_and_json(&png_bytes, &json_text).context("build ZAF container")?;

  // Write output
  let mut f = File::create(&cli.output).with_context(|| format!("create {:?}", cli.output))?;
  f.write_all(&blob)?;
  f.flush()?;

  eprintln!(
    "Built font: atlas={} json={} -> {} ({} bytes)",
    cli.atlas.display(),
    cli.json.display(),
    cli.output.display(),
    blob.len()
  );
  Ok(())
}
