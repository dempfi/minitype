// ===============================
// Cargo.toml (example)
// ===============================
// [package]
// name = "zaft"
// version = "0.3.0"
// edition = "2021"
//
// [dependencies]
// clap = { version = "4", features = ["derive"] }
// image = "0.25"
// anyhow = "1"
// thiserror = "1"
//
// # Run:
// #   cargo run --release -- stats atlas.png
// #   cargo run --release -- compress atlas.png -o atlas.zaft --bg 255 --max-pal 8
// #   cargo run --release -- decompress atlas.zaft -o roundtrip.png
//
// ===============================
// src/main.rs
// ===============================
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};

// -------------------------------
// ZAFT v3: Global-Palette + RLE + Bitpack (MCU-simple)
// -------------------------------
// Header (little-endian):
//   magic:  'Z''A''F''T' (4)
//   version:u8 (=3)
//   bg:     u8 (row-reset previous; only used by tools; decoder ignores)
//   w,h:    u32, u32
//   P:      u8  palette size in [1..16]
//   palette: P bytes (ascending by frequency)
// Payload: sequence of tokens over palette *indices* (not grayscale):
//   Token header (1 byte): bits 7..6 kind, bits 5..0 reserved(0)
//     0b00 RUN      -> [idx:u8][len:varint]          // run of index idx
//     0b01 BITPACK  -> [len:varint][packed indices]  // raw indices bit-packed at B = ceil(log2 P)
//     0b10 MIXRUN   -> [bg_idx:u8][len:varint]       // run of bg_idx (hint for typical background)
//     0b11 RESERVED
// Notes:
//   * Encoder picks a global palette of up to --max-pal (default 8) most frequent gray levels.
//   * Every pixel is mapped to nearest palette entry (by absolute difference). If P==1, whole image
//     is a single run.
//   * RUN is great for long spans of background/solid strokes. BITPACK covers the rest compactly
//     at B bits per pixel (e.g., P=8 -> 3bpp). MIXRUN is identical to RUN but lets encoder elide
//     repeating idx byte when idx==bg (we keep it explicit for simplicity in v3, see TODOs).

const MAGIC: [u8; 4] = *b"ZAFT";
const VERSION: u8 = 3;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Kind {
    Run = 0b00,
    Bitpack = 0b01,
    MixRun = 0b10,
}

#[derive(Parser)]
#[command(author, version, about = "zaft v3: global-palette RLE+bitpack for grayscale font atlases", long_about=None)]
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
        #[arg(long, default_value_t = 255)]
        bg: u8,
        #[arg(long, default_value_t = 8)]
        max_pal: u8,
    },
    Decompress {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Stats {
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compress {
            input,
            output,
            bg,
            max_pal,
        } => do_compress(input, output, bg, max_pal),
        Cmd::Decompress { input, output } => do_decompress(input, output),
        Cmd::Stats { input } => do_stats(input),
    }
}

fn do_compress(input: PathBuf, output: PathBuf, bg: u8, max_pal: u8) -> Result<()> {
    let img = image::open(&input).with_context(|| format!("open {:?}", input))?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let pixels = gray.into_raw();

    // Build global palette (top-K by frequency up to 16, limited by --max-pal)
    let k = max_pal.clamp(1, 16) as usize;
    let (palette, pcount) = build_palette(&pixels, k);
    let bpp = (32 - (pcount as u32 - 1).leading_zeros()).max(1) as u8; // ceil(log2 P)

    // Map pixels to indices
    let idxs = map_to_palette(&pixels, &palette[..pcount]);

    let payload = encode_v3(&idxs, w, h, palette[0]);

    // header
    let mut out = Vec::with_capacity(4 + 1 + 1 + 4 + 4 + 1 + pcount + payload.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(bg);
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.push(pcount as u8);
    out.extend_from_slice(&palette[..pcount]);
    out.extend_from_slice(&payload);

    let mut f = File::create(&output)?;
    f.write_all(&out)?;
    f.flush()?;

    let raw = (w as usize * h as usize) as f64;
    let ratio = raw / out.len() as f64;
    eprintln!(
        "P={pcount} (bpp={bpp}), {w}x{h} raw {} -> {} bytes, ratio {:.2}x",
        raw as usize,
        out.len(),
        ratio
    );
    Ok(())
}

fn do_decompress(input: PathBuf, output: PathBuf) -> Result<()> {
    let mut data = Vec::new();
    File::open(&input)?.read_to_end(&mut data)?;
    if data.len() < 4 + 1 + 1 + 4 + 4 + 1 {
        return Err(anyhow!("file too small"));
    }
    if data[0..4] != MAGIC {
        return Err(anyhow!("bad magic"));
    }
    if data[4] != VERSION {
        return Err(anyhow!("unsupported version {}", data[4]));
    }
    let _bg = data[5];
    let w = u32::from_le_bytes(data[6..10].try_into().unwrap());
    let h = u32::from_le_bytes(data[10..14].try_into().unwrap());
    let pcount = data[14] as usize;
    let off = 15;
    if off + pcount > data.len() {
        return Err(anyhow!("bad palette"));
    }
    let palette = &data[off..off + pcount];
    let payload = &data[off + pcount..];

    let idxs = decode_v3(payload, w, h)?;
    if idxs.iter().any(|&i| (i as usize) >= pcount) {
        return Err(anyhow!("index out of range"));
    }
    let mut px = vec![0u8; (w * h) as usize];
    for (i, &idx) in idxs.iter().enumerate() {
        px[i] = palette[idx as usize];
    }
    let img: ImageBuffer<Luma<u8>, _> =
        ImageBuffer::from_raw(w, h, px).ok_or_else(|| anyhow!("unexpected buffer size"))?;
    DynamicImage::ImageLuma8(img).save(&output)?;
    eprintln!("Decompressed -> {:?}", output);
    Ok(())
}

fn do_stats(input: PathBuf) -> Result<()> {
    let img = image::open(&input)?;
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let px = gray.into_raw();
    let mut hist = [0usize; 256];
    for &v in &px {
        hist[v as usize] += 1;
    }
    let total = (w as usize * h as usize) as f64;
    let entropy = hist
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum::<f64>();
    let uniq = hist.iter().filter(|&&c| c > 0).count();
    eprintln!("{w}x{h}, uniq={uniq}, H≈{:.2} bits/px", entropy);
    let mut top: Vec<(u8, usize)> = hist
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .map(|(i, c)| (i as u8, *c))
        .collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("Top: {:?}", &top[..top.len().min(10)]);
    Ok(())
}

// ---- palette helpers ----
fn build_palette(pixels: &[u8], k: usize) -> ([u8; 16], usize) {
    let mut hist = [0usize; 256];
    for &v in pixels {
        hist[v as usize] += 1;
    }
    let mut v: Vec<(u8, usize)> = hist
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .map(|(i, c)| (i as u8, *c))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let mut pal = [0u8; 16];
    let n = v.len().min(k.min(16));
    for i in 0..n {
        pal[i] = v[i].0;
    }
    (pal, n)
}

fn map_to_palette(pixels: &[u8], pal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len());
    for &px in pixels {
        let mut best = 0usize;
        let mut bestd = i16::MAX;
        for (i, &p) in pal.iter().enumerate() {
            let d = (px as i16 - p as i16).abs();
            if d < bestd {
                bestd = d;
                best = i;
                if d == 0 {
                    break;
                }
            }
        }
        out.push(best as u8);
    }
    out
}

// ---- encoder ----
fn encode_v3(indices: &[u8], w: u32, h: u32, bg_guess: u8) -> Vec<u8> {
    let width = w as usize;
    let mut out = Vec::with_capacity(indices.len() / 2);
    for row in 0..h as usize {
        let start = row * width;
        let end = start + width;
        let mut i = start;
        while i < end {
            let idx = indices[i];
            // try RUN >=4
            let mut j = i + 1;
            while j < end && indices[j] == idx {
                j += 1;
            }
            let run = j - i;
            if run >= 4 {
                push_hdr(&mut out, Kind::Run);
                out.push(idx);
                write_varint(&mut out, run as u32);
                i = j;
                continue;
            }
            // otherwise BITPACK a chunk until next long run
            let mut k = i + 1;
            while k < end {
                if k + 3 < end
                    && indices[k] == indices[k + 1]
                    && indices[k + 1] == indices[k + 2]
                    && indices[k + 2] == indices[k + 3]
                {
                    break;
                }
                k += 1;
            }
            push_hdr(&mut out, Kind::Bitpack);
            write_varint(&mut out, (k - i) as u32);
            bitpack_indices(&mut out, &indices[i..k]);
            i = k;
        }
    }
    out
}

fn push_hdr(out: &mut Vec<u8>, kind: Kind) {
    out.push((kind as u8) << 6);
}

fn write_varint(dst: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        dst.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    dst.push(v as u8);
}

fn bitpack_indices(dst: &mut Vec<u8>, idx: &[u8]) {
    // Determine minimal bits per index from local max (but keep simple: use global max of slice)
    let mut maxv = 0u8;
    for &x in idx {
        if x > maxv {
            maxv = x;
        }
    }
    let bits = (32 - (maxv as u32).leading_zeros()).max(1) as u8; // 1..8
    dst.push(bits); // store bits used for this block
    let mut acc: u32 = 0;
    let mut acc_bits: u8 = 0;
    for &x in idx {
        acc |= (x as u32) << acc_bits;
        acc_bits += bits;
        while acc_bits >= 8 {
            dst.push((acc & 0xFF) as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        dst.push(acc as u8);
    }
}

// ---- decoder ----
fn decode_v3(payload: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let width = w as usize;
    let mut out = vec![0u8; (w * h) as usize];
    let mut p = 0usize;
    for row in 0..h as usize {
        let mut x = 0usize;
        while x < width {
            if p >= payload.len() {
                return Err(anyhow!("eos"));
            }
            let hdr = payload[p];
            p += 1;
            let kind = match (hdr >> 6) & 0b11 {
                0 => Kind::Run,
                1 => Kind::Bitpack,
                2 => Kind::MixRun,
                _ => return Err(anyhow!("bad kind")),
            };
            match kind {
                Kind::Run | Kind::MixRun => {
                    if p >= payload.len() {
                        return Err(anyhow!("run idx missing"));
                    }
                    let idx = payload[p];
                    p += 1;
                    let (len_u, used) = read_varint(&payload[p..])?;
                    p += used;
                    for _ in 0..(len_u as usize) {
                        out[row * width + x] = idx;
                        x += 1;
                    }
                }
                Kind::Bitpack => {
                    let (len_u, used) = read_varint(&payload[p..])?;
                    p += used;
                    if p >= payload.len() {
                        return Err(anyhow!("bitpack bits missing"));
                    }
                    let bits = payload[p];
                    p += 1;
                    let mut need_bits = len_u as usize * bits as usize;
                    let mut acc: u32 = 0;
                    let mut acc_bits = 0usize;
                    let mut consumed = 0usize;
                    while consumed < len_u as usize {
                        if acc_bits < bits as usize {
                            if p >= payload.len() {
                                return Err(anyhow!("bitpack data underrun"));
                            }
                            acc |= (payload[p] as u32) << acc_bits;
                            p += 1;
                            acc_bits += 8;
                        }
                        let mask = (1u32 << bits) - 1;
                        let val = ((acc & mask) as u8);
                        acc >>= bits;
                        acc_bits -= bits as usize;
                        out[row * width + x] = val;
                        x += 1;
                        consumed += 1;
                    }
                    // ignore leftover bits in acc
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(out)
}

fn read_varint(mut buf: &[u8]) -> Result<(u32, usize)> {
    let mut shift = 0u32;
    let mut val = 0u32;
    let mut used = 0usize;
    loop {
        if buf.is_empty() {
            return Err(anyhow!("varint underflow"));
        }
        let b = buf[0];
        buf = &buf[1..];
        used += 1;
        val |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(anyhow!("varint too large"));
        }
    }
    Ok((val, used))
}
