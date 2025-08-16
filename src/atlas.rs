//! # ZAF Atlas Compression
//!
//! This module implements **ZAF** ("Zar Antialiased Font") atlas — a compact, MCU‑friendly
//! run‑length + Golomb–Rice codec for 8‑bit grayscale font atlases. It targets
//! long, horizontally arranged ASCII strips but works for any row‑major
//! L8 (grayscale) image.
//!
//! ## High‑level idea
//! 1. **Build u8 palette (16 entries)**: Compute a histogram over all 256 grayscale values and select the 16 most frequent byte values (ties break by value).
//! 2. **Nearest‑palette mapping**: Map each pixel to the palette index with the smallest absolute difference to its value.
//! 3. **RLE over palette indices**: Scan the mapped index stream and encode **runs** of identical indices.
//! 4. **Golomb–Rice lengths**: Encode the run length `L` as `Rice(L - 1; k)`,
//!    where `k` is a single global parameter chosen from the mean run length.
//! 5. **Bit‑packing (LSB‑first)**: Each run emits a 1‑bit **majority flag** and,
//!    if needed, a 4-bit palette index, followed by the Rice code.
//!
//! The result is a small header followed by a tightly packed bitstream. All I/O
//! is **LSB‑first** at the bit level to keep a tiny decoder on microcontrollers.
//!
//! ## File format (little‑endian, inner payload only)
//! The bytestream produced/consumed here omits any container magic/version. Those are
//! expected to be handled by a higher-level wrapper.
//!
//! ```text
//! Offset  Size  Field                      Notes
//! ------  ----  --------------------------  -------------------------------------
//! 0       2     width (u16, LE)
//! 2       2     height (u16, LE)
//! 4       1     k (u8)                     Global Rice parameter, 0..7
//! 5       16    palette[16] bytes          Raw u8 grayscale values
//! 21..    *     payload bits               LSB‑first bitstream (see below)
//! ```
//!
//! ### Payload bitstream per run
//! For each run of **palette indices**:
//!
//! ```text
//! [majority:1] [idx:4?] [Rice(length - 1; k)]
//! ```
//!
//! * `majority=1` means **palette index 0** (the most frequent nibble). No `idx`
//!   bits follow in that case.
//! * `majority=0` is followed by a 4‑bit **idx** (value `1..=15`).
//! * The run length is stored as Golomb–Rice code of `(L - 1)` with parameter `k`.
//!
//! All bits are written **LSB‑first** into bytes: the first bit written becomes
//! the least‑significant bit of the next output byte.
//!
//! ### Golomb–Rice coding (parameter `k`)
//! For a non‑negative integer `v`, we encode:
//!
//! ```text
//! q = v >> k              (integer division)
//! r = v & ((1 << k) - 1)  (low k bits)
//! code =  q times '1'  +  '0'  +  r in k bits (LSB‑first)
//! ```
//!
//! For `k=0`, the code degenerates to **unary**: `v` ones followed by a zero.
//!
//! ## Encoding algorithm details
//! * **Palette**: histogram over 256 bins; select 16 most frequent bytes (ties by value).
//! * **Nearest mapping**: for each pixel, find palette entry with smallest absolute difference.
//! * **Runs**: scan left‑to‑right over the whole image buffer (row‑major) and
//!   group equal **palette indices**.
//! * **Choosing `k`**: compute mean run length μ and set
//!   `k = round(log2(max(1, μ)))`, clamped to `[0,7]`.
//! * **Writing runs**: emit `[majority][idx?][Rice(len-1;k)]` per run.
//!
//! ## Decoding algorithm details
//! 1. Read and validate header, read the 16‑entry palette.
//! 2. Create a bit reader and repeatedly read runs until `w*h` pixels are
//!    produced.
//! 3. For each run, interpret the majority flag and optional 4‑bit index, then
//!    decode `len = Rice^{-1}(k) + 1`.
//! 4. Write the palette byte `palette[idx]` for `len` pixels.
//!
//! ## Error handling
//! The decoder validates:
//! * Runs never exceed the total pixel count (`run overruns output`).
//! * Payload exhaustion triggers a clean `eof` error.
//!
//! ## MCU considerations
//! * **Tiny state**: single global `k`, global palette, and bit‑level I/O.
//! * **Streaming friendly**: decoding writes runs directly into the output
//!   buffer; no need to materialize the A4 stream.
//! * **Deterministic**: no floating point at decode time; only fixed ops.
//! * **Reconstruction**: `nibble * 17` is exactly `(n << 4) | n`.
//!
//! ## Example
//! ```rust
//! # use zaf::atlas::{compress_l8, decompress_l8};
//! let w: u16 = 64; let h: u16 = 8;
//! let img = vec![0xFF; (w as usize * h as usize)];
//! let blob = compress_l8(&img, w, h);
//! let (dw, dh, out) = decompress_l8(&blob).unwrap();
//! assert_eq!((dw, dh), (w, h));
//! assert_eq!(out, img);
//! ```
//!
//! ## Format stability
//! This is the **inner ZAF payload**. If you change the header layout or payload semantics,
//! bump the outer container's version; the inner codec remains versionless.

/// Compress a row‑major **L8** (8‑bit grayscale) image into the inner ZAF
/// bytestream.
///
/// This function builds a global u8 palette (16 entries) from the most frequent
/// pixel values, maps each pixel to the nearest palette entry by absolute difference,
/// run‑length encodes palette indices, and emits run lengths using Golomb–Rice coding
/// with a single global parameter `k`. Bits are packed **LSB‑first** for a tiny MCU decoder.
///
/// # Parameters
/// * `pixels` — source pixels in row‑major order, one byte per pixel.
/// * `w`, `h` — image dimensions as `u16`. The total pixel count is `w*h`.
///
/// # Returns
/// A self‑contained **inner blob**: `header (21 bytes)` + `payload bits`.
/// The outer container (magic/version) is intentionally omitted here.
///
/// # Header layout (little‑endian)
/// ```text
/// 0..2: width  (u16)
/// 2..4: height (u16)
/// 4:    k (u8)
/// 5..21: palette[16] bytes (raw u8 values)
/// 21..: payload (LSB‑first bits)
/// ```
///
/// # Encoding of each run
/// ```text
/// [majority:1] [idx:4?] [Rice(len-1; k)]   // bits written LSB‑first
/// ```
/// * `majority=1` encodes palette index **0** without an explicit `idx`.
/// * `majority=0` is followed by a 4‑bit `idx` in `1..=15`.
///
/// # Complexity
/// * Time: `O(w*h)`
/// * Memory: `O(w*h)` transient for the palette mapping + small fixed histograms.
///
/// # Determinism
/// No floating point is used at decode. `k` is derived from the mean run
/// length at encode time using `round(log2(max(1, μ)))` clamped to `[0,7]`.
///
/// # Examples
/// ```
/// # use zaf::atlas::compress_l8;
/// let (w, h): (u16, u16) = (16, 16);
/// let img = vec![0x00; (w as usize * h as usize)];
/// let blob = compress_l8(&img, w, h);
/// assert!(blob.len() >= 21);
/// ```
pub fn compress_l8(pixels: &[u8], w: u16, h: u16) -> Vec<u8> {
  // 1) Build global 16-entry u8 palette (most frequent first) and a 256-entry map value->nearest palette idx
  let (palette, map_idx) = build_palette_u8(pixels);

  // 2) RLE over palette indices from nearest mapping
  let mut runs_std: Vec<(u8, usize)> = Vec::new(); // (palette_idx, run_len)
  let mut i = 0usize;
  while i < pixels.len() {
    let idx = map_idx[pixels[i] as usize]; // palette index of current pixel (nearest)
    let mut j = i + 1;
    while j < pixels.len() && map_idx[pixels[j] as usize] == idx {
      // extend run while same palette index
      j += 1;
    }
    runs_std.push((idx, j - i));
    i = j;
  }

  // 3) Choose global Rice parameter k from mean run length
  let mean = if runs_std.is_empty() {
    1.0
  } else {
    runs_std.iter().map(|&(_, l)| l as f64).sum::<f64>() / runs_std.len() as f64
  };
  let k: u8 = ((mean.max(1.0)).log2().round() as u8).min(7);

  // 4) Header
  let mut out = Vec::with_capacity(21 + runs_std.len() * 2);
  out.extend_from_slice(&w.to_le_bytes()); // bytes 0..2: width (u16 LE)
  out.extend_from_slice(&h.to_le_bytes()); // bytes 2..4: height (u16 LE)
  out.push(k); // byte 4: global Rice k
  // bytes 5..21: raw palette bytes
  out.extend_from_slice(&palette);

  // 5) Payload: [kind:1][idx:4?][rice(len-1;k)] … (LSB-first)
  let mut bw = BitWriter::new();
  for (idx, len) in runs_std.into_iter() {
    if idx == 0 {
      bw.write_bit(true);
    } else {
      bw.write_bit(false);
      bw.write_bits(idx as u32, 4);
    }
    write_rice(&mut bw, (len - 1) as u32, k);
  }
  bw.flush_to(&mut out);
  out
}

/// Decompress an inner ZAF bytestream into `(width, height, L8 pixels)`.
///
/// Validates palette bounds, run boundaries, and payload exhaustion. The
/// bytestream is expected to be the **inner** payload (no outer magic/version).
///
/// # Returns
/// On success, returns `(width, height, pixels)` where `pixels.len() == w*h`.
///
/// # Errors
/// * `file too small` — header shorter than 21 bytes
/// * `palette idx out of range` — decoded index ≥ 16
/// * `run overruns output` — decoded run would exceed `w*h`
/// * `eof` — payload ended prematurely while reading bits
///
/// # Bit order
/// Both the unary prefix and the `k` remainder of the Golomb–Rice code are
/// read **LSB‑first** from the payload.
///
/// # Examples
/// ```
/// # use zaf::atlas::{compress_l8, decompress_l8};
/// let (w, h): (u16, u16) = (8, 8);
/// let img = vec![0xFF; (w as usize * h as usize)];
/// let blob = compress_l8(&img, w, h);
/// let (dw, dh, out) = decompress_l8(&blob).unwrap();
/// assert_eq!((dw, dh), (w, h));
/// assert_eq!(out, img);
/// ```
pub fn decompress_l8(data: &[u8]) -> Result<(u16, u16, Vec<u8>), anyhow::Error> {
  use anyhow::anyhow;
  if data.len() < 21 {
    return Err(anyhow!("file too small"));
  }
  let w = u16::from_le_bytes([data[0], data[1]]); // width (u16 LE)
  let h = u16::from_le_bytes([data[2], data[3]]); // height (u16 LE)
  let k = data[4]; // global Rice k
  let palette_bytes = &data[5..21]; // 16 raw palette bytes
  let mut palette = [0u8; 16];
  palette.copy_from_slice(palette_bytes);

  let need = (w as usize) * (h as usize);
  let mut out = vec![0u8; need];
  let mut br = BitReader::new(&data[21..]);
  let mut x = 0usize;
  while x < need {
    let majority = br.read_bit().map_err(|_| anyhow!("eof"))?;
    let idx = if majority {
      0usize
    } else {
      br.read_bits(4).map_err(|_| anyhow!("eof"))? as usize
    };
    if idx >= 16 {
      return Err(anyhow!("palette idx out of range"));
    }
    let len = (read_rice(&mut br, k).map_err(|_| anyhow!("eof"))? + 1) as usize;
    if x + len > need {
      return Err(anyhow!("run overruns output"));
    }
    let v = palette[idx]; // write the palette byte directly
    for _ in 0..len {
      out[x] = v;
      x += 1;
    }
  }
  Ok((w, h, out))
}

/// Build a 16-entry u8 palette from the most frequent pixel values and a 256-entry map
/// from byte value -> nearest palette index (by absolute difference).
fn build_palette_u8(pixels: &[u8]) -> ([u8; 16], [u8; 256]) {
  // Histogram over 256 values
  let mut hist = [0usize; 256];
  for &v in pixels {
    hist[v as usize] += 1;
  }

  // Select top-16 values by frequency (desc), tie-break by value (asc)
  let mut items: Vec<(u8, usize)> = (0..=255).map(|v| (v as u8, hist[v])).collect();
  items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

  let mut palette = [0u8; 16];
  for i in 0..16 {
    palette[i] = items[i].0;
  }

  // Precompute nearest palette index for each possible byte (0..=255)
  let mut map_idx = [0u8; 256];
  for v in 0u16..=255 {
    let mut best_i = 0u8;
    let mut best_d = u16::MAX;
    for (i, &p) in palette.iter().enumerate() {
      let d = if v as i32 >= p as i32 {
        v as i32 - p as i32
      } else {
        p as i32 - v as i32
      } as u16;
      if d < best_d || (d == best_d && (i as u8) < best_i) {
        // tie-break by lower index
        best_d = d;
        best_i = i as u8;
      }
    }
    map_idx[v as usize] = best_i;
  }
  (palette, map_idx)
}

/// Write `Rice(val; k)` to the bitstream using LSB‑first bit order.
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

/// LSB‑first bit writer.
///
/// The first bit written becomes the LSB of the next output byte. This mirrors
/// how the decoder reads bits and simplifies very small MCU implementations that
/// prefer shift/rotate operations over masking/MSB logic.
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

/// LSB‑first bit reader for the payload section.
///
/// Pulls bytes lazily from the buffer and exposes single‑bit and fixed‑width
/// reads. Returns `Err(())` on buffer exhaustion to allow precise error mapping
/// by the caller.
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
  fn read_bit(&mut self) -> core::result::Result<bool, ()> {
    if self.bits == 0 {
      if self.pos >= self.buf.len() {
        return Err(());
      }
      self.acc = self.buf[self.pos] as u32;
      self.pos += 1;
      self.bits = 8;
    }
    let b = (self.acc & 1) != 0;
    self.acc >>= 1;
    self.bits -= 1;
    Ok(b)
  }
  fn read_bits(&mut self, n: u8) -> core::result::Result<u32, ()> {
    let mut v = 0u32;
    for i in 0..n {
      if self.read_bit()? {
        v |= 1 << i;
      }
    }
    Ok(v)
  }
}

/// Read a Golomb–Rice code with parameter `k` and return the decoded value.
fn read_rice(br: &mut BitReader, k: u8) -> core::result::Result<u32, ()> {
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

#[cfg(test)]
mod tests {
  use super::{compress_l8, decompress_l8};

  #[inline]
  fn a4(n: u8) -> u8 {
    (n << 4) | n
  }

  fn roundtrip(w: u16, h: u16, f: impl Fn(usize) -> u8) {
    let n = (w as usize) * (h as usize);
    let mut px = vec![0u8; n];
    for i in 0..n {
      px[i] = f(i);
    }
    let blob = compress_l8(&px, w, h);
    let (dw, dh, out) = decompress_l8(&blob).expect("decode");
    assert_eq!((dw, dh), (w, h));
    assert_eq!(out, px);
  }

  #[test]
  fn t_all_white() {
    roundtrip(64, 8, |_| a4(15));
  }

  #[test]
  fn t_all_black() {
    roundtrip(16, 16, |_| a4(0));
  }

  #[test]
  fn t_checker() {
    let w = 37;
    let h = 11;
    roundtrip(w, h, |i| {
      let x = i % (w as usize);
      let y = i / (w as usize);
      a4(if (x + y) % 2 == 0 { 2 } else { 12 })
    });
  }

  #[test]
  fn t_long_run_cross_rows() {
    let w = 13;
    let h = 9;
    let n = (w as usize) * (h as usize);
    roundtrip(w, h, |i| if i < n / 3 * 2 { a4(1) } else { a4(9) });
  }

  #[test]
  fn t_palette_majority() {
    let w = 40;
    let h = 3;
    let n = (w as usize) * (h as usize);
    let mut px = vec![a4(3); n];
    for i in 0..(n * 3 / 4) {
      px[i] = a4(7);
    }
    let blob = compress_l8(&px, w, h);
    let pal0 = blob[5]; // palette[0] byte in inner header
    assert_eq!(pal0, 0x77, "palette[0] should be most frequent byte");
    let (_, _, out) = decompress_l8(&blob).unwrap();
    assert_eq!(out, px);
  }
}
