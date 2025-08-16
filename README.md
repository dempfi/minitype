# ZAF Font Specification (v1)

This document defines the ZAF Font container (ZFNT) and its embedded ZAF Atlas (“inner ZAF”) payload for compact, MCU-friendly bitmap fonts with precomputed antialiasing.

- Use case: small embedded devices rendering monochrome/antialiased glyphs from a pre-baked horizontal atlas.
- Design goals: tiny decoder, deterministic parsing, fixed endianness, minimal per-glyph metadata, streaming-friendly atlas decode.

---

## 1. Conventions

- Byte order: All multi-byte integers are little-endian.
- Bit order (inner payload): All bitfields in the atlas payload are LSB-first (least significant bit written/read first).
- Units: Pixels unless stated otherwise.
- Integer types: u8, u16, u24 (3-byte little-endian), i16, u32.

---

## 2. High-level layout

A ZAF Font file is a single binary blob:

```
+--------------------+
| ZFNT Header        | 28 bytes
+--------------------+
| Glyph Table        | glyph_count × 6 bytes
+--------------------+
| Inner ZAF Atlas    | ≥ 21 bytes (see §4)
+--------------------+
```

There is no alignment padding between sections.

---

### 3. ZFNT Header (28 bytes)

| Offset | Size | Type  | Name               | Description                                               |
| ------ | ---- | ----- | ------------------ | --------------------------------------------------------- |
| 0x00   | 4    | bytes | MAGIC              | ASCII ZFNT                                                |
| 0x04   | 1    | u8    | VERSION            | File format version. Must be 1 for this spec.             |
| 0x05   | 1    | u8    | FLAGS              | Reserved, must be 0.                                      |
| 0x06   | 2    | u16   | line_height        | Baseline-to-baseline distance.                            |
| 0x08   | 2    | i16   | letter_spacing     | Additional horizontal spacing per glyph.                  |
| 0x0A   | 2    | i16   | ascent             | Pixels above baseline (non-negative recommended).         |
| 0x0C   | 2    | i16   | descent            | Pixels below baseline (non-positive recommended).         |
| 0x0E   | 2    | u16   | glyph_count        | Number of glyph records.                                  |
| 0x10   | 4    | u32   | glyph_table_offset | Absolute offset of the first glyph record (usually 0x1C). |
| 0x14   | 4    | u32   | atlas_offset       | Absolute offset of the inner ZAF atlas payload (§4).      |
| 0x18   | 4    | u32   | total_len          | Total file length in bytes (for quick bounds checks).     |

Notes

- Metrics apply to the entire font. Kerning is not modeled.
- The atlas is a single-row strip; baseline is baked in the strip and the atlas height equals ascent + (-descent) (or close); see §5.

---

## 4. Glyph Table

Immediately follows the header at glyph_table_offset. Contains glyph_count fixed-size records, 6 bytes each:

Byte(s) Type Field Range / Meaning
0..2 u24 codepoint Unicode scalar value 0..=0x10_FFFF (no surrogate values).
3..4 u16 x X-offset in the atlas strip where glyph run begins.
5 u8 w Glyph width in pixels, 0..=255.

Semantics & constraints

- x + w must be ≤ atlas.width.
- Records may appear in any order; duplicates by codepoint are invalid.
- The atlas strip is a single row; y=0 for all glyphs (baseline baked). No per-glyph y/height fields exist.

---

## 5. Inner ZAF Atlas (“ZAF” payload)

This is a self-contained grayscale atlas codec embedded at atlas_offset.

### 5.1 Inner header (21 bytes)

| Offset | Size | Type | Name     | Description                                    |
| ------ | ---- | ---- | -------- | ---------------------------------------------- |
| 0      | 2    | u16  | width    | Atlas width in pixels.                         |
| 2      | 2    | u16  | height   | Atlas height in pixels.                        |
| 4      | 1    | u8   | k Global | Golomb–Rice parameter 0..=7.                   |
| 5      | 16   | u8[] | palette  | 16 grayscale values (u8), most frequent first. |

Followed by a variable-length payload bitstream.

### 5.2 Payload bitstream (LSB-first)

The image is first mapped to a 16-entry palette of bytes (nearest by absolute difference), then run-length encoded over palette indices (0..=15) with Golomb–Rice coded lengths.

Each run is encoded as:

```
[majority:1] [idx:4?] [Rice(length-1; k)]
```

- majority = 1 encodes palette index 0 without an explicit index.
- majority = 0 is followed by a 4-bit idx value in 1..=15.
- Run length is RiceDecode(k) + 1.
- All bits are packed LSB-first into bytes.

Decoding stops after emitting width \* height pixels. Each pixel value is the palette byte at the decoded index. Output is row-major, left-to-right, top-to-bottom.

### 5.3 Encoder guidance (non-normative but canonical)

- Build a 256-bin histogram; choose the 16 most frequent bytes; tie-break by byte value ascending.
- Map each source pixel to the palette entry with smallest absolute difference; tie-break by lower palette index.
- Produce runs across the entire buffer row-major (runs may cross row boundaries).
- Set k = round(log2(max(1, mean_run_length))), clamped to [0,7].

### 5.4 Decoder validation (normative)

A compliant decoder must:

- Validate header has ≥21 bytes.
- Ensure palette array is present (16 bytes).
- Track emitted pixel count and fail on:
- Palette index ≥ 16.
- Run overruns output (emitted + len > width\*height).
- Unexpected EOF in bitstream.
- Return (width, height, pixels) where pixels.len() == width\*height.

---

## 6. Rendering Model

- The atlas is a single horizontal strip drawn with baseline baked at y=0. The atlas height represents the full ascent+descent (rounded as encoded by the producer).
- To render glyph G:
  1.  Look up its record by codepoint.
  2.  Blit rectangle [x .. x+w) × [0 .. height) from the atlas to the destination at (pen_x, pen_y - ascent), or equivalently place the atlas row such that the font baseline aligns with destination baseline.
  3.  Advance pen_x += w + letter_spacing.
      - Alpha blending uses the 8-bit value from the decoded atlas.

---

## 7. File-level Validation Rules

On parsing the outer container: 1. MAGIC == "ZFNT". 2. VERSION == 1; FLAGS == 0. 3. glyph_table_offset >= 0x1C and glyph_table_offset <= atlas_offset. 4. total_len == file_length (recommended hard check). 5. glyph_count \* 6 bytes exist starting at glyph_table_offset. 6. atlas_offset + 21 <= total_len. 7. For each glyph:

- codepoint is a valid Unicode scalar (not 0xD800..=0xDFFF).
- No duplicate codepoint.
- x + w <= atlas.width (after decoding inner header). 8. Metrics are not otherwise constrained, but typical expectations:
- line_height >= ascent - descent.
- ascent >= 0, descent <= 0 (producer discipline).

Implementations may reject files violating typical expectations.

---

## 8. Limits & Recommendations

- width, height are u16: max 65535 each.
- Glyph width w is u8: max 255.
- Prefer sorted by codepoint glyph table (not required).
- No checksum is defined; total_len plus robust decode checks are the primary integrity guard.
- To minimize RAM on MCU, decode atlas scanline by scanline if stored compressed externally; otherwise a one-shot decode into RAM is acceptable for small atlases.

---

## 9. Pseudocode

### 9.1 Parse ZFNT

```pseudo
function parse_zfnt(bytes):
  assert len(bytes) >= 0x1C
  assert bytes[0..4] == "ZFNT"
  ver = bytes[4]; flags = bytes[5]
  assert ver == 1 and flags == 0

  line_height = u16LE(bytes[0x06..0x08])
  letter_spacing = i16LE(bytes[0x08..0x0A])
  ascent = i16LE(bytes[0x0A..0x0C])
  descent = i16LE(bytes[0x0C..0x0E])
  glyph_count = u16LE(bytes[0x0E..0x10])
  gt_off = u32LE(bytes[0x10..0x14])
  atlas_off = u32LE(bytes[0x14..0x18])
  total_len = u32LE(bytes[0x18..0x1C])
  assert total_len == len(bytes)
  assert 0x1C <= gt_off <= atlas_off <= total_len

  // Glyphs
  glyphs = []
  pos = gt_off
  for i in 0..glyph_count-1:
    cp = u24LE(bytes[pos..pos+3])
    x = u16LE(bytes[pos+3..pos+5])
    w = bytes[pos+5]
    glyphs.push({cp, x, w})
    pos += 6

  // Decode inner ZAF header only to validate bounds
  (aw, ah) = read_inner_header_dimensions(bytes[atlas_off..]) // §5.1, no payload decode yet

  for g in glyphs:
    assert g.w <= 255
    assert g.x + g.w <= aw

  return {metrics, glyphs, atlas_blob: bytes[atlas_off..total_len]}
```

### 9.2 Decode inner ZAF (full)

```
function decode_inner_zaf(blob):
  assert len(blob) >= 21
  w = u16LE(blob[0..2]); h = u16LE(blob[2..4])
  k = blob[4]; palette = blob[5..21]
  out = array<u8>(w\*h)

  br = BitReaderLSB(blob[21..])
  i = 0
  while i < w*h:
    maj = br.read_bit() // EOF -> error
    idx = 0
    if not maj:
      idx = br.read_bits(4) // EOF -> error
      assert idx in 1..15
    len = rice_decode(br, k) + 1 // EOF -> error
    assert i + len <= w*h
    fill(out[i..i+len], palette[idx])
    i += len
  return (w, h, out)
```

---

## 10. Example Minimal File (illustrative)

- One glyph "A" (U+0041), width 5 at x=0.
- Metrics: line_height=16, ascent=12, descent=-4, letter_spacing=1.
- Atlas: 8×16 (numbers below are placeholders).

```

5A 46 4E 54 01 00 10 00 01 00 0C 00 FC FF 01 00
1C 00 00 00 34 00 00 00 5E 00 00 00
41 00 00 00 00 05
08 00 10 00 03 00 11 22 33 ... (16 bytes palette) ... [payload...]

```

(Spaces added for readability; not a normative sample.)

---

## 11. Producer Guidelines

- Ensure the atlas is truly single-row strip; pack glyphs tightly left-to-right without gaps where possible.
- Choose ascent, descent, and line_height to match your design; keep them consistent with the atlas height.
- Validate w <= 255; if any glyph exceeds that, split or redesign.
- Optionally store a JSON next to source assets during build; JSON is not part of ZAF but can look like:

```json
{
  "line_height": 16,
  "letter_spacing": 1,
  "ascent": 12,
  "descent": -4,
  "glyphs": [
    { "ch": "A", "x": 0, "w": 12 },
    { "ch": "a", "x": 12, "w": 11 }
  ]
}
```

The build system converts this plus a PNG into the ZAF Font file.

---

## 12. Backward/Forward Compatibility

- Files with VERSION=1 conform to this spec.
- Future versions must bump VERSION; readers should reject unknown versions unless they explicitly support them.
- FLAGS are reserved for future use and must be zero in v1 files.

---

## 13. Error Conditions (non-exhaustive)

- magic_mismatch, bad_version, bad_flags
- out_of_bounds (any section outside total_len)
- duplicate_codepoint
- codepoint_invalid (surrogate range)
- glyph_outside_atlas (x + w > atlas.width)
- inner_too_small (inner header < 21 bytes)
- inner_eof (bitstream ends early)
- inner_palette_idx_oob
- inner_run_overflow

Return a descriptive error and abort.

---

## 14. Reference Decoder Footprint (indicative)

- Outer container: parsing only.
- Inner decoder state:
- Palette: 16 bytes
- Bit reader: a few bytes of accumulator
- No floating point required.
- RAM for output buffer width\*height bytes (could stream to display if supported).
