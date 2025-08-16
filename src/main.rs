// ===============================
// Cargo.toml (example)
// ===============================
// [package]
// name = "za4f"
// version = "0.2.0"
// edition = "2021"
//
// [dependencies]
// clap = { version = "4", features = ["derive"] }
// image = "0.25"
// anyhow = "1"
//
// # Run:
// #   cargo run --release -- compress atlas.png -o atlas.za4f
// #   cargo run --release -- decompress atlas.za4f -o roundtrip.png
//
// ===============================
// src/main.rs
// ===============================
use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use std::{
  fs::File,
  io::{Read, Write},
  path::PathBuf,
};

// ---------------------------------------------
// ZA4F v2: A4 quantization + global Rice RLE + palette
// ---------------------------------------------
// Header (little-endian):
//   0..4   MAGIC  = 'Z''A''4''F'
//   4      VERSION= 2
//   5..9   width  = u32
//   9..13  height = u32
//   13     k      = u8   (Rice parameter for all lengths)
//   14..22 palette= 16 nibbles packed as 8 bytes (pal[0] in low nibble, pal[1] in high, ...)
//                 palette[0] is the MOST FREQUENT nibble value; others sorted by global frequency
// Payload (bitstream, LSB-first): sequence of runs over PALETTE INDICES
//   For each run:
//     kind:1bit    1 => majority run (palette index 0), 0 => explicit index follows
//     idx:4bits    only if kind==0 (palette index 0..15)
//     len:Rice(k)  encodes (run_length - 1)
// Decoding fills A4 values using palette[idx] (0..15), then expands to L8 as v*17 for PNG output.
// No heap required for decoding.

const MAGIC: [u8; 4] = *b"ZA4F";
const VERSION: u8 = 2;

#[derive(Parser)]
#[command(author, version, about = "za4f: Zar's Antialiased a4 Font — Rice RLE with majority palette for font atlases", long_about=None)]
struct Cli {
  #[command(subcommand)]
  cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
  Compress {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
  },
  Decompress {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
  },
}

fn main() -> Result<()> {
  let cli = Cli::parse();
  match cli.cmd {
    Cmd::Compress { input, output } => do_compress(input, output),
    Cmd::Decompress { input, output } => do_decompress(input, output),
  }
}

fn do_compress(input: PathBuf, output: PathBuf) -> Result<()> {
  let img = image::open(&input).with_context(|| format!("open {:?}", input))?;
  let gray = img.to_luma8();
  let (w, h) = gray.dimensions();
  let pixels = gray.into_raw();

  // Quantize to A4 nibbles 0..15
  let mut a4: Vec<u8> = Vec::with_capacity(pixels.len());
  for &px in &pixels {
    a4.push((px >> 4) as u8);
  }

  // Build palette by global frequency (most frequent first)
  let (palette, inv_idx) = build_palette(&a4);

  // RLE over PALETTE INDICES
  let mut runs: Vec<(u8, usize)> = Vec::new();
  let mut i = 0;
  while i < a4.len() {
    let nib = a4[i];
    let idx = inv_idx[nib as usize];
    let mut j = i + 1;
    while j < a4.len() && inv_idx[a4[j] as usize] == idx {
      j += 1;
    }
    runs.push((idx, j - i));
    i = j;
  }

  // Choose global Rice parameter k from mean run length
  let mean: f64 = runs.iter().map(|&(_, l)| l as f64).sum::<f64>() / runs.len() as f64;
  let k = ((mean.max(1.0)).log2().round() as u8).min(7);

  // Write header
  let mut out: Vec<u8> = Vec::new();
  out.extend_from_slice(&MAGIC);
  out.push(VERSION);
  out.extend_from_slice(&w.to_le_bytes());
  out.extend_from_slice(&h.to_le_bytes());
  out.push(k);
  // pack palette nibbles (16 entries) into 8 bytes
  for pair in palette.chunks(2) {
    let lo = pair[0] & 0x0F;
    let hi = pair.get(1).copied().unwrap_or(0) & 0x0F;
    out.push(lo | (hi << 4));
  }

  // Encode bitstream
  let mut bw = BitWriter::new();
  for (idx, len) in runs {
    if idx == 0 {
      // majority token
      bw.write_bit(true);
    } else {
      bw.write_bit(false);
      bw.write_bits(idx as u32, 4);
    }
    write_rice(&mut bw, (len - 1) as u32, k);
  }
  bw.flush_to(&mut out);

  let mut f = File::create(&output)?;
  f.write_all(&out)?;
  f.flush()?;
  let raw = pixels.len() as f64;
  let ratio = raw / out.len() as f64;
  eprintln!("Compressed {w}x{h} raw {} -> {} bytes, ratio {:.2}x", raw as usize, out.len(), ratio);
  Ok(())
}

fn do_decompress(input: PathBuf, output: PathBuf) -> Result<()> {
  let mut data = Vec::new();
  File::open(&input)?.read_to_end(&mut data)?;
  // Header
  if data.len() < 4 + 1 + 4 + 4 + 1 + 8 {
    return Err(anyhow!("file too small"));
  }
  if &data[0..4] != MAGIC {
    return Err(anyhow!("bad magic"));
  }
  if data[4] != VERSION {
    return Err(anyhow!("bad version"));
  }
  let w = u32::from_le_bytes(data[5..9].try_into().unwrap());
  let h = u32::from_le_bytes(data[9..13].try_into().unwrap());
  let k = data[13];
  let pal_bytes = &data[14..22];

  // Unpack palette nibbles
  let mut palette = [0u8; 16];
  for (i, b) in pal_bytes.iter().enumerate() {
    palette[i * 2] = b & 0x0F;
    palette[i * 2 + 1] = (b >> 4) & 0x0F;
  }

  let payload = &data[22..];
  let need = (w as usize) * (h as usize);

  // Decode indices and expand to L8 directly
  let mut br = BitReader::new(payload);
  let mut outpx = vec![0u8; need];
  let mut x = 0usize;
  while x < need {
    let majority = br.read_bit()?;
    let idx = if majority { 0 } else { br.read_bits(4)? as u8 } as usize;
    if idx >= 16 {
      return Err(anyhow!("palette idx out of range"));
    }
    let len = (read_rice(&mut br, k)? + 1) as usize;
    if x + len > need {
      return Err(anyhow!("run overruns output"));
    }
    let nib = palette[idx];
    let v = nib * 17; // expand A4 to L8
    for _ in 0..len {
      outpx[x] = v;
      x += 1;
    }
  }

  let img: ImageBuffer<Luma<u8>, _> = ImageBuffer::from_raw(w, h, outpx).ok_or_else(|| anyhow!("buffer size"))?;
  DynamicImage::ImageLuma8(img).save(&output)?;
  eprintln!("Decompressed -> {:?}", output);
  Ok(())
}

// ---- Palette helpers ----
fn build_palette(a4: &[u8]) -> ([u8; 16], [u8; 16]) {
  let mut hist = [0usize; 16];
  for &v in a4 {
    hist[(v & 0x0F) as usize] += 1;
  }
  // produce list of (nib, count) sorted by count desc then nib asc
  let mut items: [(u8, usize); 16] = [(0, 0); 16];
  for nib in 0..16 {
    items[nib] = (nib as u8, hist[nib]);
  }
  items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  // palette: order of nibbles; inv_idx: nibble->palette index
  let mut palette = [0u8; 16];
  let mut inv = [0u8; 16];
  for (i, (nib, _)) in items.iter().enumerate() {
    palette[i] = *nib & 0x0F;
    inv[*nib as usize] = i as u8;
  }
  (palette, inv)
}

// ---- Rice coding ----
fn write_rice(bw: &mut BitWriter, val: u32, k: u8) {
  let q = if k == 0 { val } else { val >> k };
  let r = if k == 0 { 0 } else { val & ((1 << k) - 1) };
  for _ in 0..q {
    bw.write_bit(true);
  }
  bw.write_bit(false);
  if k > 0 {
    bw.write_bits(r, k);
  }
}
fn read_rice(br: &mut BitReader, k: u8) -> Result<u32> {
  let mut q = 0u32;
  while br.read_bit()? {
    q += 1;
  }
  if k == 0 {
    Ok(q)
  } else {
    let r = br.read_bits(k)?;
    Ok((q << k) | r)
  }
}

// ---- Bit IO (LSB-first) ----
struct BitWriter {
  acc: u32,
  bits: u8,
  buf: Vec<u8>,
}
impl BitWriter {
  fn new() -> Self {
    Self { acc: 0, bits: 0, buf: Vec::new() }
  }
  fn write_bit(&mut self, b: bool) {
    self.acc |= (b as u32) << self.bits;
    self.bits += 1;
    if self.bits >= 8 {
      self.buf.push(self.acc as u8);
      self.acc >>= 8;
      self.bits -= 8;
    }
  }
  fn write_bits(&mut self, mut v: u32, n: u8) {
    for _ in 0..n {
      self.write_bit((v & 1) != 0);
      v >>= 1;
    }
  }
  fn flush_to(mut self, out: &mut Vec<u8>) {
    if self.bits > 0 {
      self.buf.push(self.acc as u8);
    }
    out.extend_from_slice(&self.buf);
  }
}

struct BitReader<'a> {
  buf: &'a [u8],
  pos: usize,
  acc: u32,
  bits: u8,
}
impl<'a> BitReader<'a> {
  fn new(buf: &'a [u8]) -> Self {
    Self { buf, pos: 0, acc: 0, bits: 0 }
  }
  fn read_bit(&mut self) -> Result<bool> {
    if self.bits == 0 {
      if self.pos >= self.buf.len() {
        return Err(anyhow!("eof"));
      }
      self.acc = self.buf[self.pos] as u32;
      self.pos += 1;
      self.bits = 8;
    }
    let b = self.acc & 1;
    self.acc >>= 1;
    self.bits -= 1;
    Ok(b != 0)
  }
  fn read_bits(&mut self, n: u8) -> Result<u32> {
    let mut v = 0u32;
    for i in 0..n {
      if self.read_bit()? {
        v |= 1 << i;
      }
    }
    Ok(v)
  }
}
