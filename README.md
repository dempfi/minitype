# ZAF Font Specification

This document defines the **ZAF Font container (ZFNT)** and its embedded **ZAF Atlas** (the “inner ZAF”) payload for compact, MCU‑friendly bitmap fonts with precomputed antialiasing.

- **Use case:** small embedded devices rendering monochrome/antialiased glyphs from a pre‑baked horizontal atlas.
- **Design goals:** tiny decoder, deterministic parsing, fixed endianness, minimal per‑glyph metadata, streaming‑friendly atlas decode.

---

## 1. Conventions

- **Byte order:** Little‑endian for all multi‑byte integers.
- **Bit order (inner payload):** LSB‑first (least significant bit written/read first).
- **Units:** Pixels unless stated otherwise.
- **Integer types:** `u8`, `u16`, `u24` (3‑byte little‑endian), `i16`, `u32`.

---

## 2. High‑level layout

A ZFNT file is one binary blob composed of:

```
+--------------------+
| ZFNT Header        | 46 bytes fixed + 7×charset_seg_count
+--------------------+
| Charset Segments   | charset_seg_count × 7 bytes (inside header region)
+--------------------+
| Glyph Table        | glyph_count × 4 bytes
+--------------------+
| Inner ZAF Atlas    | ≥ 21 bytes (see §5)
+--------------------+
| Kerning Pairs      | kerning_count × 7 bytes (optional)
+--------------------+
```

There is no padding between sections. Offsets and lengths are recorded in the header so readers can seek directly.

---

## 3. ZFNT Header (variable length)

The header begins with a 46‑byte **fixed portion**, immediately followed by `charset_seg_count` charset segment records (7 bytes each). After the segments comes the glyph table.

| Offset | Size | Type  | Name                   | Description                                           |
| -----: | ---: | :---- | ---------------------- | ----------------------------------------------------- |
|   0x00 |    4 | bytes | **MAGIC**              | ASCII `ZFNT`                                          |
|   0x04 |    1 | u8    | **VERSION**            | File format version. **Must be 1** for this spec.     |
|   0x05 |    1 | u8    | **FLAGS**              | Reserved = 0                                          |
|   0x06 |    2 | u16   | **line_height**        | Baseline‑to‑baseline distance.                        |
|   0x08 |    2 | i16   | **letter_spacing**     | Additional horizontal spacing per glyph.              |
|   0x0A |    2 | i16   | **ascent**             | Pixels above baseline (non‑negative recommended).     |
|   0x0C |    2 | i16   | **descent**            | Pixels below baseline (non‑positive recommended).     |
|   0x0E |    2 | u16   | **glyph_count**        | Number of glyph records.                              |
|   0x10 |    4 | u32   | **glyph_table_offset** | Absolute offset of the first glyph record.            |
|   0x14 |    4 | u32   | **glyph_table_len**    | Total bytes of the glyph table (== 4 × glyph_count).  |
|   0x18 |    4 | u32   | **atlas_offset**       | Absolute offset of the inner ZAF atlas payload (§5).  |
|   0x1C |    4 | u32   | **atlas_len**          | Total bytes of the inner ZAF atlas payload.           |
|   0x20 |    4 | u32   | **total_len**          | Total file length in bytes (for quick bounds checks). |
|   0x24 |    4 | u32   | **kerning_offset**     | Absolute offset of kerning block, or 0 if none.       |
|   0x28 |    4 | u32   | **kerning_count**      | Number of kerning pairs; each is 7 bytes (see §6).    |
|   0x2C |    2 | u16   | **charset_seg_count**  | Number of charset segments that immediately follow.   |
|   0x2E |  7×N | —     | **charset_segments**   | N = charset_seg_count; see **§4** for record format.  |

> **Note:** The header length is `0x2E + 7×charset_seg_count`. The **glyph table** begins exactly at `glyph_table_offset`, which equals that header length.

---

## 4. Charset Segments (always present)

ZFNT v3 **always** uses a charset‑based mapping; per‑glyph Unicode codepoints are not stored in the glyph table. Instead, a compact segment table maps **contiguous** Unicode ranges to sequential glyph indices.

Each charset segment record is 7 bytes:

| Bytes | Type | Field      | Meaning                                                         |
| :---: | :--: | ---------- | --------------------------------------------------------------- |
| 0..2  | u24  | start_cp   | First Unicode scalar value in the range (inclusive).            |
| 2..5  | u16  | len        | Number of codepoints in this segment (≥1).                      |
| 5..7  | u16  | glyph_base | Glyph index of `start_cp`. Next codepoints follow sequentially. |

**Total covered codepoints** across all segments **must equal** `glyph_count`.

**Runtime lookup:** For codepoint `cp`, find the segment where `start_cp ≤ cp < start_cp + len`, then:

```
index = glyph_base + (cp - start_cp)
```

If no segment matches, the glyph is not covered by this font; a renderer may use a fallback font.

---

## 5. Glyph Table (4 bytes / glyph)

Immediately after the charset segments. Contains `glyph_count` fixed‑size records, 4 bytes each:

| Bytes | Type | Field   | Range / Meaning                                     |
| :---: | :--: | ------- | --------------------------------------------------- |
| 0..2  | u16  | x       | X‑offset in the atlas strip where glyph run begins. |
|   2   |  u8  | w       | Glyph width in pixels (0..=255).                    |
|   3   |  i8  | advance | Signed horizontal advance for this glyph.           |

**Constraints**

- `x + w ≤ atlas.width` (validated after decoding the inner header).
- There is no per‑glyph `y` or `height`; the atlas is a single horizontal strip whose height equals the font’s ascent + |descent|.

---

## 6. Kerning Block (optional)

If present (`kerning_count > 0` and `kerning_offset != 0`), the kerning block is a sequence of 7‑byte pairs:

| Bytes | Type | Field | Meaning                                                     |
| :---: | :--: | ----- | ----------------------------------------------------------- |
| 0..3  | u24  | left  | Left codepoint (Unicode scalar).                            |
| 3..6  | u24  | right | Right codepoint (Unicode scalar).                           |
|   6   |  i8  | adj   | Signed adjustment applied to advance when this pair occurs. |

Pairs are unordered; producers may store them sorted for binary search.

---

## 7. Inner ZAF Atlas (embedded image codec)

The atlas carries a compact grayscale image with a tiny decoder. It begins at `atlas_offset` and has length `atlas_len`.

### 7.1 Inner header (21 bytes)

```
Offset  Size  Field                  Notes
0       2     width (u16, LE)
2       2     height (u16, LE)
4       1     k                      Global Rice parameter, 0..7
5       16    palette[16] (u8)       Raw grayscale values
21..    *     payload bits           LSB‑first bitstream (see 7.2)
```

### 7.2 Payload bitstream (LSB‑first)

The image is first mapped to a 16‑entry byte palette, then run‑length encoded over palette indices (0..15) with Golomb–Rice coded lengths. Majority index 0 is implicit for 1‑bit savings per run.

Decoding stops after emitting `width × height` pixels. Output is row‑major.

---

## 8. Rendering Model

- The atlas is a single horizontal strip. Its height corresponds to **ascent + |descent|**. The strip’s baseline is implied by these metrics; producers typically place the baseline `ceil(ascent)` rows from the top.
- To render codepoint **cp**:
  1. Map **cp → glyph index** via the charset segments (see §4).
  2. Load glyph `(x, w, advance)` at that index.
  3. Blit rectangle `[x .. x+w) × [0 .. atlas.height)` from the atlas to destination at `(pen_x, pen_y - ascent)` so baselines align.
  4. Apply kerning if a pair `(prev_cp, cp)` exists: `pen_x += adj`.
  5. Advance: `pen_x += advance + letter_spacing`.
- **Blend:** Use the atlas byte as straight alpha when compositing to the destination.

---

## 9. File‑level Validation Rules

A compliant reader should:

1. Check `MAGIC == "ZFNT"` and `VERSION == 1`.
2. Ensure `total_len == file_length`.
3. Ensure `glyph_table_offset ≥ 0x2E` and `glyph_table_offset == 0x2E + 7×charset_seg_count`.
4. Ensure `glyph_table_len == 4 × glyph_count`.
5. Ensure `atlas_offset == glyph_table_offset + glyph_table_len` and `atlas_offset + 21 ≤ total_len`.
6. If `kerning_offset != 0`: ensure `kerning_offset ≥ atlas_offset + atlas_len` and space for `7 × kerning_count` exists.
7. Decode only the inner header (7.1) to get `atlas.width/height`, then ensure for each glyph that `x + w ≤ atlas.width`.
8. Metric sanity (recommended but not mandatory): `line_height ≥ ascent - descent`, `ascent ≥ 0`, `descent ≤ 0`.

---

## 10. Example JSON (producer‑side sidecar)

This JSON is **not** part of ZFNT; it is used by the reference builder. With charset mode, glyphs omit codepoints entirely.

```json
{
  "line_height": 16,
  "letter_spacing": 1,
  "ascent": 12,
  "descent": -4,
  "charset": [{ "start": " ", "end": "~" }],
  "glyphs": [
    { "x": 0, "w": 5 },
    { "x": 7, "w": 6 }
  ],
  "kerning": [{ "left": "A", "right": "V", "adj": -2 }]
}
```

---

## 11. Producer Guidelines

- **Charset mode only:** Per‑glyph codepoints are not serialized; define contiguous ranges that cover your glyph set.
- **Advances:** Store per‑glyph `advance` (i8). If omitted in sidecar metadata, the builder uses `advance = w`.
- **Kerning:** Optional. Include only meaningful pairs to save space.
- **Atlas strip:** Pack glyphs left‑to‑right. A small gap (e.g., 2 px) between glyphs helps avoid sampling bleed.
- **Integrity:** No checksum field; rely on `total_len` and robust decoder checks.
