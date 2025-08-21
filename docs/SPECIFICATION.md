# MiniType Font Specification

Compact, MCU-friendly bitmap font container with tight atlas packing and per-glyph quantization hints.

- **Use case:** embedded targets rendering antialiased glyphs from a pre-baked horizontal atlas strip.
- **Goals:** tiny, deterministic decoder; fixed endianness; minimal metadata; streaming-friendly O(1) seeks.

---

## 1. Conventions

- **Endianness:** Little-endian for all multi-byte integers.
- **Units:** Pixels unless stated otherwise.
- **Integer types:** `u8`, `u16`, `u24` (3 bytes LE), `i16`, `u32`.
- **L4 packing:** 2 pixels per byte — **low nibble = left pixel**, **high nibble = right pixel**.

---

## 2. File layout (high-level)

```
+––––––––––----------+
| Header             | 0x2C fixed + 7×charset_seg_count
+–––––----------–––––+
| Charset Segments   | inside header region (7 bytes each)
+––––––––----------––+
| Glyph Table        | glyph_count × 6 bytes
+–––----------–––––––+
| Atlas (inner)      | see §7
+–––––––––----------–+
| Kerning (optional) | kerning_count × 7 bytes
+––––––––––----------+
```

No padding. Offsets recorded in the header.

---

## 3. Header (fixed + variable)

Fixed 44 bytes, then `segment_count` charset records.

| Off  | Size | Type | Field         | Description                   |
| :--: | :--: | :--: | ------------- | ----------------------------- |
| 0x00 |  4   |      | MAGIC         | ASCII `"MFNT"`                |
| 0x04 |  1   |  u8  | VERSION       | **1**                         |
| 0x05 |  1   |  u8  | FLAGS         | Reserved = 0                  |
| 0x06 |  2   | u16  | line_height   | Baseline-to-baseline          |
| 0x08 |  2   | i16  | ascent        | Pixels above baseline         |
| 0x0A |  2   | i16  | descent       | Pixels below baseline         |
| 0x0C |  2   | u16  | glyph_count   | Number of glyphs              |
| 0x0E |  4   | u32  | glyphs_off    | Offset to glyph table         |
| 0x12 |  4   | u32  | glyphs_len    | **6 × glyph_count**           |
| 0x16 |  4   | u32  | atlas_off     | Offset to inner atlas         |
| 0x1A |  4   | u32  | atlas_len     | Length of inner atlas payload |
| 0x1E |  4   | u32  | total_len     | Entire file length            |
| 0x22 |  4   | u32  | kerning_off   | Kerning block offset or 0     |
| 0x26 |  4   | u32  | kerning_cnt   | Number of kerning pairs       |
| 0x2A |  2   | u16  | segment_count | Charset segment count         |
| 0x2C | 7×N  |      | charset_segs  | N = segment_count (see §4)    |

---

## 4. Charset segments (7 bytes each)

| Bytes | Type | Field    | Meaning                   |
| :---: | :--: | -------- | ------------------------- |
| 0..3  | u24  | start_cp | First codepoint           |
| 3..5  | u16  | len      | Number of codepoints      |
| 5..7  | u16  | base     | Glyph index of `start_cp` |

Maps Unicode ranges → glyph indices.

---

## 5. Glyph table (5 bytes per glyph)

| Bytes | Type | Field   | Description             |
| :---: | :--: | ------- | ----------------------- |
| 0..2  | u16  | x       | X offset in atlas strip |
|   2   |  u8  | w       | Glyph width (px)        |
|   3   |  i8  | advance | Signed advance          |
|   4   |  i8  | left    | Left side bearing       |

---

## 6. Kerning block (optional)

If `kerning_cnt > 0`:

| Bytes | Type | Field | Meaning         |
| :---: | :--: | ----- | --------------- |
| 0..3  | u24  | left  | Left codepoint  |
| 3..6  | u24  | right | Right codepoint |
|   6   |  i8  | adj   | Advance adjust  |

---

## 7. Inner atlas (tight-crop)

### 7.1 Atlas header (70 bytes)

| Off | Size | Type | Field   | Notes                     |
| :-: | :--: | :--: | ------- | ------------------------- |
|  0  |  2   | u16  | width   | Atlas width (px)          |
|  2  |  2   | u16  | h_tight | Height of serialized band |
|  4  |  2   | u16  | y_off   | Offset from original top  |

### 7.2 Row payload

- Immediately follows header.
- `row_stride = ceil(width/2)` bytes.
- Total = `h_tight × row_stride`.
- Pixels stored as linear L4 indices: low nibble=left, high=right.
- Odd width → last high nibble = 0, ignored.

---

## 8. Rendering model

1. Map codepoint → glyph idx (§4).
2. Get glyph `(x, w, advance, left, q)` (§5).
3. Rectangle in atlas: `[x..x+w) × [0..h_tight)`.
4. Iterate row-major:
   - Row ptr = `payload + ry*row_stride + (x>>1)`
   - Use `(x&1)` for nibble select.
   - Expand nibble → alpha `nibble * 16`.
5. Blit at `(pen_x+left, pen_y-ascent+y_off)`.
6. Apply kerning if present.
7. Advance pen_x.

---

## 9. Validation rules

- MAGIC = `"MFNT"`, VERSION=1
- `glyphs_len == 6×glyph_count`
- `atlas_len == 70 + h_tight×row_stride`
- `x+w ≤ width` for all glyphs
- If kerning: `kerning_off ≥ atlas_off+atlas_len` and fits in file.
