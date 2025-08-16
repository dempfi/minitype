use clap::Parser;
use image::{DynamicImage, GrayImage, ImageReader, Luma};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

// OI (8bpp grayscale) — container unchanged, bitstream v3 (BREAKING)
// ------------------------------------------------------------------
// Header (big-endian):
//   magic:  b"oi00"
//   width:  u32
//   height: u32
//   cs:     u8  (0 for grayscale)
//   pad:    [u8;3] = 0
//
// Per row:
//   predictor: 2 bits  (00 NONE, 01 SUB, 10 UP, 11 AVG)
//   v3 token stream (MSB-first) producing exactly `width` residuals.
//
// Opcodes:
//   0                        : ZRUN — gamma(L)          (L>=1)         [prev=0 afterwards]
//   10                       : RUN  — gamma(L)          (L>=1)         [repeat prev residual]
//   110 dddd                 : DIFF4  (d in [-8..7])    (store d+8)
//   1110 dddddd              : DIFF6  (d in [-32..31])  (store d+32)
//   11110 gamma(L)+L bytes   : LITERAL (no byte align inside)
//   111110 dddddddd gamma(L) : COPY (within current row residuals), distance=1..255, L>=3
//
// Notes:
//   * Only final stream is byte-aligned; rows/literals aren’t.
//   * COPY is on residuals, not final pixels.

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

    let encoded = encode_oi_v3(&pixels, w, h);

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
    fn write_u8(&mut self, b: u8) {
        self.write(b as u32, 8);
    }
    fn write_gamma(&mut self, n: u32) {
        debug_assert!(n >= 1);
        let b = 32 - n.leading_zeros(); // bits in n
        for _ in 1..b {
            self.write(0, 1);
        } // (b-1) zeros
        self.write(1, 1); // the 1
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

// ===================== Predictors & helpers =====================
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
                    let r = row[x].wrapping_sub(avg);
                    out.push(r);
                    left = row[x];
                }
            } else {
                // no prev row: SUB-like is a good fallback
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

#[inline]
fn gamma_bits(n: u32) -> usize {
    let b = 32 - n.leading_zeros();
    (2 * b - 1) as usize
}

// estimator (COPY not modeled; encoder still uses COPY when beneficial)
fn count_bits_row_est(res: &[u8]) -> usize {
    let mut bits = 0usize;
    let mut prev = 0u8;
    let mut i = 0usize;
    while i < res.len() {
        if res[i] == 0 {
            let mut l = 0usize;
            while i < res.len() && res[i] == 0 {
                l += 1;
                i += 1;
            }
            bits += 1 + gamma_bits(l as u32);
            prev = 0;
            continue;
        }
        if res[i] == prev {
            let mut l = 0usize;
            while i < res.len() && res[i] == prev {
                l += 1;
                i += 1;
            }
            bits += 2 + gamma_bits(l as u32);
            continue;
        }
        let b = res[i];
        let sp = if prev < 128 {
            prev as i16
        } else {
            prev as i16 - 256
        };
        let sb = if b < 128 { b as i16 } else { b as i16 - 256 };
        let d = sb - sp;
        if (-8..=7).contains(&d) {
            bits += 3 + 4;
            prev = b;
            i += 1;
            continue;
        }
        if (-32..=31).contains(&d) {
            bits += 4 + 6;
            prev = b;
            i += 1;
            continue;
        }
        // literal
        let start = i;
        let mut l = 1usize;
        i += 1;
        prev = b;
        while i < res.len() {
            let nb = res[i];
            if nb == 0 || nb == prev {
                break;
            }
            let snb = if nb < 128 { nb as i16 } else { nb as i16 - 256 };
            let d2 = snb
                - (if prev < 128 {
                    prev as i16
                } else {
                    prev as i16 - 256
                });
            if (-8..=7).contains(&d2) || (-32..=31).contains(&d2) {
                break;
            }
            prev = nb;
            i += 1;
            l += 1;
        }
        bits += 5 + gamma_bits(l as u32) + 8 * l;
    }
    bits
}

// ===================== Encoder v3 =====================
pub fn encode_oi_v3(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + pixels.len() / 2 + 64);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&w.to_be_bytes());
    out.extend_from_slice(&h.to_be_bytes());
    out.push(0);
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

        let cand = [
            (Pred::None, make_residuals(Pred::None, row, prev)),
            (Pred::Sub, make_residuals(Pred::Sub, row, prev)),
            (Pred::Up, make_residuals(Pred::Up, row, prev)),
            (Pred::Avg, make_residuals(Pred::Avg, row, prev)),
        ];
        let (best_pred, best_res) = cand
            .iter()
            .map(|(p, r)| (*p, r.as_slice(), count_bits_row_est(r)))
            .min_by_key(|(_, _, bits)| *bits)
            .map(|(p, r, _)| (p, r.to_vec()))
            .unwrap();

        match best_pred {
            Pred::None => bw.write(0b00, 2),
            Pred::Sub => bw.write(0b01, 2),
            Pred::Up => bw.write(0b10, 2),
            Pred::Avg => bw.write(0b11, 2),
        }

        encode_row_v3_into(&mut bw, &best_res);
        prev_row = Some(row.to_vec());
    }

    bw.byte_align();
    out.extend_from_slice(&bw.finish());
    out.extend_from_slice(&OI_END);
    out
}

// Try COPY (within row) at i
fn find_best_copy(res: &[u8], i: usize) -> Option<(usize /*dist*/, usize /*len*/)> {
    if i == 0 {
        return None;
    }
    let max_back = i.min(255);
    let max_len = (res.len() - i).min(64);
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    for d in 1..=max_back {
        let mut l = 0usize;
        while l < max_len && res[i + l] == res[i + l - d] {
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

fn encode_row_v3_into(bw: &mut BitWriter, res: &[u8]) {
    let mut prev = 0u8;
    let mut i = 0usize;
    while i < res.len() {
        // ZRUN
        if res[i] == 0 {
            let mut l = 0usize;
            while i < res.len() && res[i] == 0 {
                l += 1;
                i += 1;
            }
            bw.write(0b0, 1);
            bw.write_gamma(l as u32);
            prev = 0;
            continue;
        }
        // RUN(prev)
        if res[i] == prev {
            let mut l = 0usize;
            while i < res.len() && res[i] == prev {
                l += 1;
                i += 1;
            }
            bw.write(0b10, 2);
            bw.write_gamma(l as u32);
            continue;
        }

        // COPY: before diffs/literal
        if let Some((dist, len)) = find_best_copy(res, i) {
            let copy_bits = 6 + 8 + gamma_bits(len as u32);
            let lit_bits = 5 + gamma_bits(len as u32) + 8 * len;
            if copy_bits < lit_bits {
                bw.write(0b111110, 6);
                bw.write(dist as u32, 8);
                bw.write_gamma(len as u32);
                let end = i + len;
                while i < end {
                    let v = res[i - dist];
                    prev = v;
                    i += 1;
                }
                continue;
            }
        }

        // DIFF4 / DIFF6
        let b = res[i];
        let sp = if prev < 128 {
            prev as i16
        } else {
            prev as i16 - 256
        };
        let sb = if b < 128 { b as i16 } else { b as i16 - 256 };
        let d = sb - sp;
        if (-8..=7).contains(&d) {
            bw.write(0b110, 3);
            bw.write((d + 8) as u32, 4);
            prev = b;
            i += 1;
            continue;
        }
        if (-32..=31).contains(&d) {
            bw.write(0b1110, 4);
            bw.write((d + 32) as u32, 6);
            prev = b;
            i += 1;
            continue;
        }

        // LITERAL
        let start = i;
        let mut l = 1usize;
        i += 1;
        prev = b;
        while i < res.len() {
            let nb = res[i];
            if nb == 0 || nb == prev {
                break;
            }
            let snb = if nb < 128 { nb as i16 } else { nb as i16 - 256 };
            let d2 = snb
                - (if prev < 128 {
                    prev as i16
                } else {
                    prev as i16 - 256
                });
            if (-8..=7).contains(&d2) || (-32..=31).contains(&d2) {
                break;
            }
            if let Some((_d, mlen)) = find_best_copy(res, i) {
                if mlen >= 4 {
                    break;
                }
            }
            prev = nb;
            i += 1;
            l += 1;
        }
        bw.write(0b11110, 5);
        bw.write_gamma(l as u32);
        for &v in &res[start..start + l] {
            bw.write_u8(v);
        }
    }
}

// ===================== Decoder v3 =====================
pub fn decode_oi_v3(data: &[u8]) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    if data.len() < 16 || &data[0..4] != MAGIC.as_slice() {
        anyhow::bail!("Not an OI file (bad magic)");
    }
    let w = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let h = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

    let mut i = 16usize;
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let mut br = BitReader::new(&data[i..]);

    let width = w as usize;
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

        // decode residuals
        let mut res: Vec<u8> = Vec::with_capacity(width);
        let mut prev = 0u8;
        while res.len() < width {
            let b1 = br.read_bit()?;
            if b1 == 0 {
                let l = br.read_gamma()? as usize;
                res.extend(std::iter::repeat(0u8).take(l));
                prev = 0;
                continue;
            }
            let b2 = br.read_bit()?;
            if b2 == 0 {
                let l = br.read_gamma()? as usize;
                res.extend(std::iter::repeat(prev).take(l));
                continue;
            }
            let b3 = br.read_bit()?;
            if b3 == 0 {
                let d = br.read(4)? as i16 - 8;
                let v = prev.wrapping_add(d as i8 as u8);
                res.push(v);
                prev = v;
                continue;
            }
            let b4 = br.read_bit()?;
            if b4 == 0 {
                let d = br.read(6)? as i16 - 32;
                let v = prev.wrapping_add(d as i8 as u8);
                res.push(v);
                prev = v;
                continue;
            }
            let b5 = br.read_bit()?;
            if b5 == 0 {
                // LITERAL: prefix 11110
                let l = br.read_gamma()? as usize;
                for _ in 0..l {
                    let v = br.read_u8()?;
                    res.push(v);
                    prev = v;
                }
                continue;
            }
            // COPY: prefix must be 111110 (consume the sixth bit = 0)
            let b6 = br.read_bit()?;
            if b6 != 0 {
                anyhow::bail!("Reserved/unknown opcode (expected COPY 111110)");
            }
            let dist = br.read(8)? as usize;
            let l = br.read_gamma()? as usize;
            if dist == 0 || dist > res.len() {
                anyhow::bail!("Bad COPY distance");
            }
            for _ in 0..l {
                let v = res[res.len() - dist];
                res.push(v);
                prev = v;
            }
        }

        // reconstruct pixels
        match pred {
            Pred::None => {
                out.extend_from_slice(&res);
                prev_row_out = Some(res);
            }
            Pred::Sub => {
                let mut row_out = vec![0u8; width];
                let mut left = 0u8;
                for x in 0..width {
                    let px = res[x].wrapping_add(left);
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
                        row_out[x] = res[x].wrapping_add(prev_row[x]);
                    }
                } else {
                    row_out.copy_from_slice(&res);
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
                        let px = res[x].wrapping_add(avg);
                        row_out[x] = px;
                        left = px;
                    }
                } else {
                    let mut left = 0u8;
                    for x in 0..width {
                        let px = res[x].wrapping_add(left);
                        row_out[x] = px;
                        left = px;
                    }
                }
                out.extend_from_slice(&row_out);
                prev_row_out = Some(row_out);
            }
        }
    }

    i += br.byte;
    br.align_byte();
    if i + 8 <= data.len() && data[i..i + 8] == OI_END {
        Ok((w, h, out))
    } else {
        anyhow::bail!("Missing END marker")
    }
}

// ===================== Tests =====================
#[cfg(test)]
mod tests {
    use super::*;

    // Simple deterministic RNG (xorshift32)
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

    // replace the old roundtrip with this safer one
    fn roundtrip(pixels: &[u8], w: u32, h: u32) {
        assert_eq!(
            pixels.len(),
            (w as usize) * (h as usize),
            "pixels.len() must equal w*h"
        );
        let enc = encode_oi_v3(pixels, w, h);
        assert!(enc.len() >= 16 + 8);
        assert_eq!(&enc[0..4], &MAGIC);
        let (dw, dh, out) = decode_oi_v3(&enc).expect("decode");
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(out, pixels);
        assert_eq!(&enc[enc.len() - 8..], &OI_END);
    }

    #[test]
    fn small_row_opcodes() {
        // Construct a row that triggers: ZRUN, RUN, DIFF4, DIFF6, LITERAL, COPY.
        // Row width 32, single row image.
        let mut row = vec![];
        // ZRUN of 6 zeros
        row.extend_from_slice(&[0; 6]);
        // RUN(prev): repeat previous residual (set prev by a literal first)
        row.push(10); // literal-ish to seed prev
        row.extend_from_slice(&[10, 10, 10, 10]); // RUN prev
        // DIFF4 around prev 10 -> 13 (+3) and 7 (-3)
        row.extend_from_slice(&[13, 7]);
        // DIFF6 larger step: from 7 to 7+25=32
        row.push(32);
        // LITERAL noise
        row.extend_from_slice(&[200, 55, 90, 123]);
        // COPY within row: repeat pattern "ABCD" we just wrote
        row.extend_from_slice(&[200, 55, 90, 123]);

        let w = row.len() as u32;
        roundtrip(&row, w, 1);
    }

    #[test]
    fn predictors_patterns() {
        // NONE: flat ramp in each row but changing per row slowly (may pick NONE or AVG)
        let (w, h) = (32u32, 8u32);
        let mut img = vec![0u8; (w * h) as usize];
        for y in 0..h {
            let base = (y * 3) as u8;
            for x in 0..w {
                img[(y * w + x) as usize] = base.wrapping_add(x as u8);
            }
        }
        roundtrip(&img, w, h);

        // SUB: pure horizontal ramps (same row repeated), SUB should be strong
        let (w2, h2) = (64u32, 4u32);
        let mut img2 = vec![0u8; (w2 * h2) as usize];
        for y in 0..h2 {
            let mut v = (y * 5) as u8;
            for x in 0..w2 {
                img2[(y * w2 + x) as usize] = v;
                v = v.wrapping_add(1);
            }
        }
        roundtrip(&img2, w2, h2);

        // UP: vertical gradient (same column changes, rows are repeated)
        let (w3, h3) = (16u32, 32u32);
        let mut img3 = vec![0u8; (w3 * h3) as usize];
        for y in 0..h3 {
            for x in 0..w3 {
                img3[(y * w3 + x) as usize] = (y as u8) * 4;
            }
        }
        roundtrip(&img3, w3, h3);

        // AVG: checkerboard-ish smooth averaging helps a bit
        let (w4, h4) = (32u32, 16u32);
        let mut img4 = vec![0u8; (w4 * h4) as usize];
        for y in 0..h4 {
            for x in 0..w4 {
                let up = if y > 0 {
                    img4[((y - 1) * w4 + x) as usize]
                } else {
                    0
                };
                let left = if x > 0 {
                    img4[(y * w4 + x - 1) as usize]
                } else {
                    0
                };
                let avg = (((up as u16 + left as u16) / 2) as u8).wrapping_add(((x ^ y) as u8) & 1);
                img4[(y * w4 + x) as usize] = avg;
            }
        }
        roundtrip(&img4, w4, h4);
    }

    #[test]
    fn copy_catches_repeats() {
        // Build a single row with repeated tile to trigger COPY
        let tile: [u8; 8] = [5, 5, 6, 6, 7, 7, 6, 6];
        let mut row = Vec::new();
        // prefix noise to avoid pure ZRUN/RUN
        row.extend_from_slice(&[9, 11, 9, 11]); // +4
        for _ in 0..6 {
            row.extend_from_slice(&tile);
        } // +48
        row.extend_from_slice(&[33, 44, 55, 66]); // +4  => total 56

        let w = row.len() as u32; // <-- fix: width must equal row length
        let h = 1;
        roundtrip(&row, w, h);
    }

    #[test]
    fn random_images_small_medium() {
        let mut rng = Rng::new(0xC0FFEEu32);
        for &(w, h) in &[(1, 1), (2, 3), (7, 5), (16, 16), (31, 9), (64, 32)] {
            let mut img = vec![0u8; (w * h) as usize];
            rng.fill(&mut img);
            roundtrip(&img, w, h);
        }
    }

    #[test]
    fn random_large() {
        let (w, h) = (128u32, 96u32);
        let mut rng = Rng::new(0xDEADBEEF);
        let mut img = vec![0u8; (w * h) as usize];
        rng.fill(&mut img);
        roundtrip(&img, w, h);
    }

    #[test]
    fn container_and_end_marker() {
        let w = 10;
        let h = 3;
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i * 7 % 256) as u8).collect();
        let enc = encode_oi_v3(&pixels, w, h);
        assert_eq!(&enc[0..4], &MAGIC);
        let (dw, dh, out) = decode_oi_v3(&enc).unwrap();
        assert_eq!((dw, dh), (w, h));
        assert_eq!(out, pixels);
        assert_eq!(&enc[enc.len() - 8..], &OI_END);
    }
}
