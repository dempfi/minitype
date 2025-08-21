# [UNSTABLE] MiniType Font Specification

Compact, MCU-friendly bitmap font container with tight atlas packing and per-glyph quantization hints.

- **Use case:** embedded targets rendering antialiased glyphs from a pre-baked horizontal atlas strip.
- **Goals:** tiny, deterministic decoder; fixed endianness; minimal metadata; streaming-friendly O(1) seeks.

## 1. Conventions

- **Endianness:** Little-endian for all multi-byte integers.
- **Units:** Pixels unless stated otherwise.
- **Integer types:** `u8`, `u16`, `u24` (3 bytes LE), `i16`, `u32`.
- **L4 packing:** 2 pixels per byte — **low nibble = left pixel**, **high nibble = right pixel**.

## 2. File layout (high-level)

```
+---------------------+
| Header              | 14 bytes + 7 × segment_count
+---------------------+
| Charset Segments    | inside header region (7 bytes each)
+---------------------+
| Glyph Table         | glyph_count × 6 bytes (bit-packed)
+---------------------+
| Atlas Blob          | concatenation of per-glyph tight rows
+---------------------+
| Kerning (optional)  | kerning_count × 7 bytes
+---------------------+
```

The kerning block is present only if there are bytes remaining after the atlas blob: `u16 kerning_count, then kerning_count × 5-byte pairs`

## 3. Header (fixed + variable)

Fixed 14 bytes, then `segment_count` charset records.

| Off  | Size | Type | Field         | Description           |
| :--: | :--: | :--: | ------------- | --------------------- |
| 0x00 |  4   |      | MAGIC         | ASCII `"MFNT"`        |
| 0x04 |  1   |  u8  | VERSION       | **0**                 |
| 0x05 |  1   |  u8  | FLAGS         | Reserved = 0          |
| 0x06 |  1   |  u8  | line_height   | Baseline-to-baseline  |
| 0x07 |  1   |  i8  | ascent        | Pixels above baseline |
| 0x08 |  1   |  i8  | descent       | Pixels below baseline |
| 0x09 |  1   |  u8  | segment_count | Charset segment count |

Immediately following the prefix:

1. Segments: segment_count × 5 bytes (see §4)
2. Glyph count: u16 glyph_count
3. Glyph table: glyph_count × 6 bytes (see §5)
4. Atlas blob: variable length (see §6)
5. Kerning tail (optional): u16 kerning_count then kerning_count × 5 bytes (see §7)

## 4. Charset segments (5 bytes each)

| Bytes | Type | Field    | Meaning              |
| :---: | :--: | -------- | -------------------- |
| 0..3  | u24  | start_cp | First codepoint      |
| 3..5  | u16  | len      | Number of codepoints |

## 5. Glyph table (6 bytes per glyph)

|  Bits   | Type | Field           | Description                                     |
| :-----: | :--: | --------------- | ----------------------------------------------- |
|  0..=6  |  u7  | w               | Glyph width (px)                                |
| 7..=13  |  u8  | w               | Glyph width (px)                                |
| 14..=20 |  u8  | y_off_first_row | Offset from line top to glyph's first tight row |
| 21..=25 |  i5  | advance_delta   | (advance - w), range -16..+15                   |
| 26..=29 |  i4  | left_bearing    | Left side bearing                               |
| 30..=47 | u18  | glyph_blob_off  | Byte offset from start of atlas blob            |

Derived:

```
advance = (w as i16) + (advance_delta as i16)
bytes_per_row = ceil(w / 2)
glyph_blob_span = bytes_per_row * h (bytes)
glyph_blob_ptr = file[atlas_off + glyph_blob_off..]
```

## 6. Atlas blob (per-glyph tight rows)

The atlas blob is a concatenation of tightly cropped L4 rows for each glyph, in glyph order.

For each glyph:

- Rows: exactly h rows.
- Each row has `bytes_per_row = ceil(w/2)` bytes.
- Order: row 0, row 1, … row h-1, then the next glyph's rows.

L4 rows: low nibble = left pixel, high nibble = right pixel; odd width → last high nibble is padding and ignored.

## 7. Kerning tail (optional; 5 bytes per pair)

If present:

| Bytes | Type | Field | Meaning           |
| :---: | :--: | ----- | ----------------- |
| 0..2  | u16  | left  | Left glyph index  |
| 2..4  | u16  | right | Right glyph index |
|   4   |  i8  | adj   | Advance adjust    |

## 8. Rendering model

1. Compute line_top relative to the baseline using ascent/descent.
2. Map codepoint → glyph index via segments (§4).
3. Read glyph record (§5) → `{w, h, yoff, left_bearing, adv_delta, blob_off}`.
4. `advance = w + adv_delta`.
5. Seek pixels: `ptr = atlas_off + blob_off`.
6. For each row 0..h-1, read `bytes_per_row` bytes, expand L4 nibbles to coverage (e.g., `alpha = nibble << 4`), and blit at:
   - `dst_x = pen_x + left_bearing`
   - `dst_y = line_top + yoff + row`
7. Advance pen: `pen_x += advance`, then apply kerning if available.

## 9. Validation rules

- MAGIC = `"MFNT"`, VERSION=0
- `file_len ≥ header_len + glyph_table_len + kerning_len`
- `kerning_off = file_len - 7*kerning_count` and `kerning_off ≥ atlas_off`.
  • Charset segments must be consistent and cover exactly glyph_count entries when bases are applied.
