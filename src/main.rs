use clap::Parser;
use image::{DynamicImage, GrayImage, ImageReader, Luma};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

// OI (8bpp grayscale) — lean stream (ZRUN, LITERAL, COPY) with per-row predictors
// -------------------------------------------------------------------------------
// Header (big-endian):
//   magic:  b"oi00"
//   width:  u32
//   height: u32
//   cs:     u8  (0 = this stream)
//   pad:    [u8;3] = 0
//
// For each row:
//   predictor: 2 bits  (00 NONE, 01 SUB, 10 UP, 11 AVG)
//   row bitstream (MSB-first), producing exactly `width` bytes
//
// Opcodes:
//   0                        : ZRUN  — gamma(L)          (L>=1) emit L zero bytes
//   10                       : LIT   — gamma(L) + L bytes (raw)
//   110 dddddddd gamma(L)    : COPY  — distance=1..255 from already produced bytes in this row; L>=3
//
// Notes:
//   * Only the final stream is byte-aligned; rows/opcodes are bit-packed.
//   * COPY works within the current row residual/pixel stream (post-predictor space).

const MAGIC: [u8; 4] = *b"oi00";
const OI_END: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

#[derive(Parser, Debug)]
#[command(name = "oi", version)]
struct Args {
    input: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input_path = Path::new(&args.input);
    let img = ImageReader::open(&input_path)?
        .with_guessed_format()?
        .decode()?;
    let gray = to_grayscale(img);
    let (w, h) = gray.dimensions();

    let mut pixels: Vec<u8> = Vec::with_capacity((w as usize) * (h as usize));
    for y in 0..h {
        for x in 0..w {
            pixels.push(gray.get_pixel(x, y)[0]);
        }
    }

    let encoded = encode_oi(&pixels, w, h);

    let out_path = input_path.with_extension("oi");
    let mut f = BufWriter::new(File::create(&out_path)?);
    f.write_all(&encoded)?;
    f.flush()?;
    eprintln!(
        "Wrote {} ({}x{}, {} bytes)",
        out_path.display(),
        w,
        h,
        encoded.len()
    );
    Ok(())
}

fn to_grayscale(img: DynamicImage) -> GrayImage {
    match img {
        DynamicImage::ImageLuma8(g) => g,
        DynamicImage::ImageLumaA8(ga) => {
            let (w, h) = ga.dimensions();
            let mut out = GrayImage::new(w, h);
            for (x, y, p) in ga.enumerate_pixels() {
                out.put_pixel(x, y, Luma([p.0[0]]));
            }
            out
        }
        _ => img.to_luma8(),
    }
}

// ===================== Bit I/O =====================
struct BitWriter {
    buf: Vec<u8>,
    acc: u32,
    nbits: u8,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    fn write(&mut self, bits: u32, n: u8) {
        let mut v = bits & ((1u32 << n) - 1);
        let mut k = n;
        while k > 0 {
            let space = 8 - self.nbits;
            let take = space.min(k);
            let shift = k - take;
            let chunk = (v >> shift) & ((1 << take) - 1);
            self.acc = (self.acc << take) | chunk;
            self.nbits += take;
            k -= take;
            if self.nbits == 8 {
                self.buf.push(self.acc as u8);
                self.acc = 0;
                self.nbits = 0;
            }
        }
    }
    fn write_bit(&mut self, b: bool) {
        self.write(b as u32, 1);
    }
    fn write_u8(&mut self, b: u8) {
        self.write(b as u32, 8);
    }
    fn write_gamma(&mut self, n: u32) {
        // Elias gamma code for n>=1
        debug_assert!(n >= 1);
        let b = 32 - n.leading_zeros();
        for _ in 1..b {
            self.write(0, 1);
        } // (b-1) zeros
        self.write(1, 1);
        if b > 1 {
            self.write(n & ((1 << (b - 1)) - 1), (b - 1) as u8);
        }
    }
    fn byte_align(&mut self) {
        if self.nbits != 0 {
            self.write(0, 8 - self.nbits);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits != 0 {
            self.buf.push((self.acc << (8 - self.nbits)) as u8);
        }
        self.buf
    }
}

struct BitReader<'a> {
    s: &'a [u8],
    byte: usize,
    acc: u32,
    nbits: u8,
}
impl<'a> BitReader<'a> {
    fn new(s: &'a [u8]) -> Self {
        Self {
            s,
            byte: 0,
            acc: 0,
            nbits: 0,
        }
    }
    fn read(&mut self, n: u8) -> anyhow::Result<u32> {
        while self.nbits < n {
            if self.byte >= self.s.len() {
                anyhow::bail!("Bitstream truncated");
            }
            self.acc = (self.acc << 8) | self.s[self.byte] as u32;
            self.byte += 1;
            self.nbits += 8;
        }
        let shift = self.nbits - n;
        let mask = (1u32 << n) - 1;
        let bits = (self.acc >> shift) & mask;
        self.nbits -= n;
        self.acc &= (1u32 << self.nbits) - 1;
        Ok(bits)
    }
    fn read_bit(&mut self) -> anyhow::Result<u32> {
        self.read(1)
    }
    fn read_u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.read(8)? as u8)
    }
    fn read_gamma(&mut self) -> anyhow::Result<u32> {
        // gamma decode: zeros until first 1, then read payload of that many bits-1
        let mut z = 0u32;
        loop {
            let b = self.read_bit()?;
            if b == 0 {
                z += 1;
            } else {
                break;
            }
        }
        let payload = if z == 0 { 0 } else { self.read(z as u8)? };
        Ok((1u32 << z) | payload)
    }
    fn align_byte(&mut self) {
        let _ = if self.nbits % 8 != 0 {
            self.read(self.nbits % 8)
        } else {
            Ok(0)
        };
    }
}

// ===================== Predictors & residuals =====================
#[derive(Copy, Clone, Debug)]
enum Pred {
    None,
    Sub,
    Up,
    Avg,
}

fn make_residuals(pred: Pred, row: &[u8], prev_row: Option<&[u8]>) -> Vec<u8> {
    let w = row.len();
    let mut out = Vec::with_capacity(w);
    match pred {
        Pred::None => out.extend_from_slice(row),
        Pred::Sub => {
            let mut left = 0u8;
            for &v in row {
                let r = v.wrapping_sub(left);
                out.push(r);
                left = v;
            }
        }
        Pred::Up => {
            if let Some(p) = prev_row {
                for x in 0..w {
                    out.push(row[x].wrapping_sub(p[x]));
                }
            } else {
                out.extend_from_slice(row);
            }
        }
        Pred::Avg => {
            if let Some(p) = prev_row {
                let mut left = 0u8;
                for x in 0..w {
                    let up = p[x];
                    let avg = (((left as u16 + up as u16) >> 1) & 0xFF) as u8;
                    out.push(row[x].wrapping_sub(avg));
                    left = row[x];
                }
            } else {
                // first row: SUB-like
                let mut left = 0u8;
                for &v in row {
                    let r = v.wrapping_sub(left);
                    out.push(r);
                    left = v;
                }
            }
        }
    }
    out
}

// Cheap row cost estimator for choosing predictor (matches our 3-opcode set)
#[inline]
fn gamma_bits(n: u32) -> usize {
    let b = 32 - n.leading_zeros();
    (2 * b - 1) as usize
}

fn estimate_row_bits(res: &[u8]) -> usize {
    let mut bits = 0usize;
    let mut i = 0usize;
    while i < res.len() {
        if res[i] == 0 {
            let mut l = 0usize;
            while i < res.len() && res[i] == 0 {
                l += 1;
                i += 1;
            }
            bits += 1 + gamma_bits(l as u32); // ZRUN: "0" + gamma(L)
            continue;
        }
        // COPY estimate
        let mut best_len = 0usize;
        let max_back = i.min(255);
        let max_len = (res.len() - i).min(255);
        for d in 1..=max_back {
            let mut l = 0usize;
            while i + l < res.len() && res[i + l] == res[i + l - d] && l < max_len {
                l += 1;
            }
            if l >= 3 && l > best_len {
                best_len = l;
                if l == max_len {
                    break;
                }
            }
        }
        if best_len >= 3 {
            bits += 3 + 8 + gamma_bits(best_len as u32); // prefix 110 + dist + gamma(L)
            i += best_len;
            continue;
        }
        // LITERAL group until a compressible event
        let start = i;
        let mut l = 1usize;
        i += 1;
        while i < res.len() {
            if res[i] == 0 {
                break;
            }
            // if COPY likely soon, stop literal early
            let look_copy = {
                let max_back = i.min(255);
                let max_len = (res.len() - i).min(255);
                let mut m = 0usize;
                for d in 1..=max_back {
                    let mut ll = 0usize;
                    while i + ll < res.len() && res[i + ll] == res[i + ll - d] && ll < max_len {
                        ll += 1;
                    }
                    if ll > m {
                        m = ll;
                        if m >= 4 {
                            break;
                        }
                    }
                }
                m >= 4
            };
            if look_copy {
                break;
            }
            i += 1;
            l += 1;
        }
        let _ = &res[start..start + l];
        bits += 2 + gamma_bits(l as u32) + 8 * l; // "10" + gamma(L) + L bytes
    }
    bits
}

// ===================== Encoder =====================
pub fn encode_oi(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    debug_assert_eq!(pixels.len(), (w as usize) * (h as usize));
    let mut out = Vec::with_capacity(16 + pixels.len() / 2 + 64);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&w.to_be_bytes());
    out.extend_from_slice(&h.to_be_bytes());
    out.push(0); // cs=0
    out.extend_from_slice(&[0, 0, 0]);

    if pixels.is_empty() {
        out.extend_from_slice(&OI_END);
        return out;
    }

    let width = w as usize;
    let mut bw = BitWriter::new();
    let mut prev_row: Option<Vec<u8>> = None;

    for row_idx in 0..(h as usize) {
        let row = &pixels[row_idx * width..(row_idx + 1) * width];
        let prev = prev_row.as_deref();

        // choose predictor by estimated bit cost
        let cand = [
            (Pred::None, make_residuals(Pred::None, row, prev)),
            (Pred::Sub, make_residuals(Pred::Sub, row, prev)),
            (Pred::Up, make_residuals(Pred::Up, row, prev)),
            (Pred::Avg, make_residuals(Pred::Avg, row, prev)),
        ];
        let (best_pred, res) = cand
            .iter()
            .map(|(p, r)| (*p, r.as_slice(), estimate_row_bits(r)))
            .min_by_key(|(_, _, bits)| *bits)
            .map(|(p, r, _)| (p, r.to_vec()))
            .unwrap();

        match best_pred {
            Pred::None => bw.write(0b00, 2),
            Pred::Sub => bw.write(0b01, 2),
            Pred::Up => bw.write(0b10, 2),
            Pred::Avg => bw.write(0b11, 2),
        }

        encode_row_into(&mut bw, &res);
        prev_row = Some(row.to_vec());
    }

    bw.byte_align();
    out.extend_from_slice(&bw.finish());
    out.extend_from_slice(&OI_END);
    out
}

fn find_best_copy(res: &[u8], i: usize) -> Option<(usize /*dist*/, usize /*len*/)> {
    if i == 0 {
        return None;
    }
    let max_back = i.min(255);
    let max_len = (res.len() - i).min(255);
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    for d in 1..=max_back {
        let mut l = 0usize;
        while i + l < res.len() && res[i + l] == res[i + l - d] && l < max_len {
            l += 1;
        }
        if l >= 3 && l > best_len {
            best_len = l;
            best_dist = d;
            if l == max_len {
                break;
            }
        }
    }
    if best_len >= 3 {
        Some((best_dist, best_len))
    } else {
        None
    }
}

fn encode_row_into(bw: &mut BitWriter, res: &[u8]) {
    let mut i = 0usize;
    while i < res.len() {
        // ZRUN
        if res[i] == 0 {
            let mut l = 1usize;
            while i + l < res.len() && res[i + l] == 0 {
                l += 1;
            }
            bw.write_bit(false); // "0"
            bw.write_gamma(l as u32);
            i += l;
            continue;
        }

        // COPY first (beats literal if worthwhile)
        if let Some((dist, len)) = find_best_copy(res, i) {
            let copy_bits = 3 + 8 + gamma_bits(len as u32);
            let lit_bits = 2 + gamma_bits(len as u32) + 8 * len;
            if copy_bits < lit_bits {
                bw.write(0b110, 3);
                bw.write(dist as u32, 8);
                bw.write_gamma(len as u32);
                i += len;
                continue;
            }
        }

        // LITERAL: grow until a compressible event
        let start = i;
        let mut l = 1usize;
        i += 1;
        while i < res.len() {
            if res[i] == 0 {
                break;
            }
            if let Some((_d, mlen)) = find_best_copy(res, i) {
                if mlen >= 4 {
                    break;
                }
            }
            i += 1;
            l += 1;
        }
        bw.write(0b10, 2);
        bw.write_gamma(l as u32);
        for &v in &res[start..start + l] {
            bw.write_u8(v);
        }
    }
}

// ===================== Decoder =====================
pub fn decode_oi(data: &[u8]) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    if data.len() < 16 || &data[0..4] != MAGIC.as_slice() {
        anyhow::bail!("Not an OI file (bad magic)");
    }
    let w = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let h = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    // cs = data[12]; // currently only 0
    let mut br = BitReader::new(&data[16..]);
    let width = w as usize;
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let mut prev_row_out: Option<Vec<u8>> = None;

    for _row in 0..h {
        let pred_bits = br.read(2)? as u8;
        let pred = match pred_bits {
            0b00 => Pred::None,
            0b01 => Pred::Sub,
            0b10 => Pred::Up,
            0b11 => Pred::Avg,
            _ => unreachable!(),
        };

        let mut row: Vec<u8> = Vec::with_capacity(width);
        while row.len() < width {
            let b1 = br.read_bit()?;
            if b1 == 0 {
                // ZRUN
                let l = br.read_gamma()? as usize;
                row.extend(std::iter::repeat(0u8).take(l));
            } else {
                let b2 = br.read_bit()?;
                if b2 == 0 {
                    // LITERAL
                    let l = br.read_gamma()? as usize;
                    for _ in 0..l {
                        row.push(br.read_u8()?);
                    }
                } else {
                    // COPY (prefix 110: we've already seen '11', now ensure the next bit is 0)
                    let b3 = br.read_bit()?;
                    if b3 != 0 {
                        anyhow::bail!("Reserved/unknown opcode (expected COPY 110)");
                    }
                    let dist = br.read(8)? as usize;
                    let l = br.read_gamma()? as usize;
                    if dist == 0 || dist > row.len() {
                        anyhow::bail!("Bad COPY distance");
                    }
                    for _ in 0..l {
                        let v = row[row.len() - dist];
                        row.push(v);
                    }
                }
            }
        }
        // Reconstruct with predictor
        match pred {
            Pred::None => {
                out.extend_from_slice(&row);
                prev_row_out = Some(row);
            }
            Pred::Sub => {
                let mut row_out = vec![0u8; width];
                let mut left = 0u8;
                for x in 0..width {
                    let px = row[x].wrapping_add(left);
                    row_out[x] = px;
                    left = px;
                }
                out.extend_from_slice(&row_out);
                prev_row_out = Some(row_out);
            }
            Pred::Up => {
                let mut row_out = vec![0u8; width];
                if let Some(prev_row) = prev_row_out.as_ref() {
                    for x in 0..width {
                        row_out[x] = row[x].wrapping_add(prev_row[x]);
                    }
                } else {
                    row_out.copy_from_slice(&row);
                }
                out.extend_from_slice(&row_out);
                prev_row_out = Some(row_out);
            }
            Pred::Avg => {
                let mut row_out = vec![0u8; width];
                if let Some(prev_row) = prev_row_out.as_ref() {
                    let mut left = 0u8;
                    for x in 0..width {
                        let up = prev_row[x];
                        let avg = (((left as u16 + up as u16) >> 1) & 0xFF) as u8;
                        let px = row[x].wrapping_add(avg);
                        row_out[x] = px;
                        left = px;
                    }
                } else {
                    let mut left = 0u8;
                    for x in 0..width {
                        let px = row[x].wrapping_add(left);
                        row_out[x] = px;
                        left = px;
                    }
                }
                out.extend_from_slice(&row_out);
                prev_row_out = Some(row_out);
            }
        }
    }

    // Align to the next byte, then compute consumed bytes precisely
    br.align_byte();
    let consumed = 16 + br.byte; // bytes read from the slice (after alignment)

    if consumed + 8 > data.len() {
        anyhow::bail!("Truncated after rows");
    }
    if &data[consumed..consumed + 8] != &OI_END {
        anyhow::bail!("Missing END marker");
    }

    Ok((w, h, out))
}

// ===================== Tests =====================
#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic RNG
    struct Rng(u32);
    impl Rng {
        fn new(seed: u32) -> Self {
            Rng(seed)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        fn next_u8(&mut self) -> u8 {
            (self.next_u32() & 0xFF) as u8
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf {
                *b = self.next_u8();
            }
        }
    }

    fn roundtrip(pixels: &[u8], w: u32, h: u32) {
        assert_eq!(
            pixels.len(),
            (w as usize) * (h as usize),
            "pixels.len() must equal w*h"
        );
        let enc = encode_oi(pixels, w, h);
        assert_eq!(&enc[0..4], &MAGIC);
        let (dw, dh, dec) = decode_oi(&enc).expect("decode");
        assert_eq!((dw, dh), (w, h));
        assert_eq!(dec, pixels);
        assert_eq!(&enc[enc.len() - 8..], &OI_END);
    }

    #[test]
    fn small_opcodes_and_copy() {
        // Single row exercising ZRUN, LITERAL, COPY
        let mut row = vec![];
        row.extend_from_slice(&[0; 7]); // ZRUN
        row.extend_from_slice(&[10, 11, 12, 13]); // LITERAL
        row.extend_from_slice(&[10, 11, 12, 13]); // COPY candidate
        roundtrip(&row, row.len() as u32, 1);
    }

    #[test]
    fn predictors_patterns() {
        // SUB-strong rows
        let (w, h) = (64u32, 4u32);
        let mut img = vec![0u8; (w * h) as usize];
        for y in 0..h {
            let mut v = (y * 5) as u8;
            for x in 0..w {
                img[(y * w + x) as usize] = v;
                v = v.wrapping_add(1);
            }
        }
        roundtrip(&img, w, h);

        // UP-strong: vertical gradient
        let (w2, h2) = (16u32, 32u32);
        let mut img2 = vec![0u8; (w2 * h2) as usize];
        for y in 0..h2 {
            for x in 0..w2 {
                img2[(y * w2 + x) as usize] = (y as u8) * 3;
            }
        }
        roundtrip(&img2, w2, h2);

        // AVG-ish checker
        let (w3, h3) = (32u32, 16u32);
        let mut img3 = vec![0u8; (w3 * h3) as usize];
        for y in 0..h3 {
            for x in 0..w3 {
                let up = if y > 0 {
                    img3[((y - 1) * w3 + x) as usize]
                } else {
                    0
                };
                let left = if x > 0 {
                    img3[(y * w3 + x - 1) as usize]
                } else {
                    0
                };
                let avg = (((up as u16 + left as u16) / 2) as u8).wrapping_add(((x ^ y) as u8) & 1);
                img3[(y * w3 + x) as usize] = avg;
            }
        }
        roundtrip(&img3, w3, h3);
    }

    #[test]
    fn copy_repeats() {
        let tile: [u8; 8] = [5, 5, 6, 6, 7, 7, 6, 6];
        let mut row = Vec::new();
        row.extend_from_slice(&[9, 11, 9, 11]);
        for _ in 0..6 {
            row.extend_from_slice(&tile);
        }
        row.extend_from_slice(&[33, 44, 55, 66]);
        roundtrip(&row, row.len() as u32, 1);
    }

    #[test]
    fn random_images() {
        let mut rng = Rng::new(0xC0FFEE);
        for &(w, h) in &[(1, 1), (3, 2), (7, 5), (16, 16), (31, 9), (64, 32)] {
            let mut img = vec![0u8; (w * h) as usize];
            rng.fill(&mut img);
            roundtrip(&img, w, h);
        }
    }

    #[test]
    fn container_and_end_marker() {
        let w = 10;
        let h = 3;
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i * 7 % 256) as u8).collect();
        let enc = encode_oi(&pixels, w, h);
        assert_eq!(&enc[0..4], &MAGIC);
        assert_eq!(&enc[enc.len() - 8..], &OI_END);
        let (dw, dh, out) = decode_oi(&enc).unwrap();
        assert_eq!((dw, dh), (w, h));
        assert_eq!(out, pixels);
    }
}
