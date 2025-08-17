# MiniType Font Specification

This document defines the **MiniType Font container** and its embedded atlas payload for compact, MCU‑friendly bitmap fonts with precomputed antialiasing.

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

A MiniType file is one binary blob composed of:

```
+--------------------+
| Header             | 44 bytes fixed + 7×charset_seg_count
+--------------------+
| Charset Segments   | charset_seg_count × 7 bytes (inside header region)
+--------------------+
| Glyph Table        | glyph_count × 4 bytes
+--------------------+
| Inner MiniType Atlas | ≥ 20 + ceil(height/8) bytes (see §7)
+--------------------+
| Kerning Pairs      | kerning_count × 7 bytes (optional)
+--------------------+
```

There is no padding between sections. Offsets and lengths are recorded in the header so readers can seek directly.

**Inner MiniType sizing:** Let `rm = ceil(height/8)`. The inner atlas size equals `20 + rm + payload_bytes`, where `payload_bytes = present_row_count * ceil(width/2)` and `present_row_count` is the number of rows with row_mask bit = 1.

---

## 3. Header (variable length)

The header begins with a 44‑byte **fixed portion**, immediately followed by `charset_seg_count` charset segment records (7 bytes each). After the segments comes the glyph table.

| Offset | Size | Type  | Name                   | Description                                               |
| -----: | ---: | :---- | ---------------------- | --------------------------------------------------------- |
|   0x00 |    4 | bytes | **MAGIC**              | ASCII `MFNT`                                              |
|   0x04 |    1 | u8    | **VERSION**            | File format version. **Must be 1** for this spec.         |
|   0x05 |    1 | u8    | **FLAGS**              | Reserved = 0                                              |
|   0x06 |    2 | u16   | **line_height**        | Baseline‑to‑baseline distance.                            |
|   0x08 |    2 | i16   | **ascent**             | Pixels above baseline (non‑negative recommended).         |
|   0x0A |    2 | i16   | **descent**            | Pixels below baseline (non‑positive recommended).         |
|   0x0C |    2 | u16   | **glyph_count**        | Number of glyph records.                                  |
|   0x0E |    4 | u32   | **glyph_table_offset** | Absolute offset of the first glyph record.                |
|   0x12 |    4 | u32   | **glyph_table_len**    | Total bytes of the glyph table (== 4 × glyph_count).      |
|   0x16 |    4 | u32   | **atlas_offset**       | Absolute offset of the inner MiniType atlas payload (§7). |
|   0x1A |    4 | u32   | **atlas_len**          | Total bytes of the inner MiniType atlas payload.          |
|   0x1E |    4 | u32   | **total_len**          | Total file length in bytes (for quick bounds checks).     |
|   0x22 |    4 | u32   | **kerning_offset**     | Absolute offset of kerning block, or 0 if none.           |
|   0x26 |    4 | u32   | **kerning_count**      | Number of kerning pairs; each is 7 bytes (see §6).        |
|   0x2A |    2 | u16   | **charset_seg_count**  | Number of charset segments that immediately follow.       |
|   0x2C |  7×N | -     | **charset_segments**   | N = charset_seg_count; see **§4** for record format.      |

> **Note:** The header length is `0x2C + 7×charset_seg_count`. The **glyph table** begins exactly at `glyph_table_offset`, which equals that header length.

---

## 4. Charset Segments

MiniType stores a compact segment table mapping **contiguous** Unicode ranges to sequential glyph indices.

Each charset segment record is 7 bytes:

| Bytes | Type | Field      | Meaning                                                         |
| :---: | :--: | ---------- | --------------------------------------------------------------- |
| 0..3  | u24  | start_cp   | First Unicode scalar value in the range (inclusive).            |
| 3..5  | u16  | len        | Number of codepoints in this segment (≥1).                      |
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

- `x + w ≤ atlas.width` (validated after decoding the inner header in §7.1).

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

## 7. Atlas

The atlas is stored as a compact grayscale image intended for tiny MCU decoders. **Version 1** uses a fixed **L4 indices + 16‑entry palette** with **row skipping**. All decoding operations are byte/nibble only—no bit‑level arithmetic or floating point.

### 7.1 Inner header (variable length)

| Offset                  | Size             | Field            | Notes                                               |
| ----------------------- | ---------------- | ---------------- | --------------------------------------------------- |
| 0                       | 2                | width (u16, LE)  |                                                     |
| 2                       | 2                | height (u16, LE) |                                                     |
| 4                       | 16               | palette[16] (u8) | palette[0] MUST be 0x00 (black)                     |
| 20                      | ceil(height / 8) | row_mask         | (bitset, LSB-first; 1=row present, 0=row all black) |
| (20 + ceil(height / 8)) | \*               | payload          | concatenated packed rows for rows with mask=1       |

Where `rm = ceil(height / 8)`.

- **Palette semantics:** `palette[0]` is black (0x00). Producers SHOULD choose the remaining 15 entries as the most frequent **nonzero** grayscale values in the image, sorted by descending frequency (tie‑break by lower value). Consumers must treat the palette as opaque byte values.

### 7.2 Payload layout (packed L4 indices)

- Pixels are encoded as **indices (0..15)** into `palette`.
- Two pixels per byte: **low nibble = left pixel**, **high nibble = right pixel**.
- For odd widths, the final byte’s **high nibble MUST be zero** and MUST be ignored by readers.
- Rows whose **row_mask bit = 0** have **no payload bytes** and are implicitly filled with `palette[0]`.

**Note:** Nibble order is fixed as stated above. The “LSB‑first” bit order in §1 applies **only** to the bits inside `row_mask`, not to the nibble packing of pixels.

**Row order:** Row‑major, top‑to‑bottom. For each row `y` from `0..height-1`:

1. Read the bit `b = ((row_mask[y / 8] >> (y % 8)) & 1)`.
2. If `b == 0`, emit `width` pixels of `palette[0]`.
3. If `b == 1`, read `ceil(width / 2)` bytes and expand nibbles to palette indices.

### 7.3 Decoder notes (MCU)

- Only byte/nibble ops; no variable‑length bitstream.
- Constant‑time index→value via 16‑byte palette.
- Streaming friendly: to blit a glyph rect `[x..x+w) × [0..height)`, a reader may skip whole black rows cheaply via `row_mask`, then consume exactly `ceil(width/2)` bytes for present rows, seeking to the needed columns.

---

## 8. Rendering Model

- The atlas is a single horizontal strip. Its height corresponds to **ascent + |descent|**. The strip’s baseline is implied by these metrics; producers typically place the baseline `ceil(ascent)` rows from the top.
- To render codepoint **cp**:

  1. Map **cp → glyph index** via the charset segments (see §4).
  2. Load glyph `(x, w, advance)` at that index.
  3. Blit rectangle `[x .. x+w) × [0 .. atlas.height)` from the atlas to destination at `(pen_x, pen_y - ascent)` so baselines align.
  4. Apply kerning if a pair `(prev_cp, cp)` exists: `pen_x += adj`.
  5. Advance: `pen_x += advance.`

- **Blend:** After palette lookup, use the resulting byte value as straight alpha when compositing to the destination.

---

## 9. File‑level Validation Rules

A compliant reader should:

1. Check `MAGIC == "MFNT"` and `VERSION == 1`.
2. Ensure `total_len == file_length`.
3. Ensure `glyph_table_offset ≥ 0x2C` and `glyph_table_offset == 0x2C + 7×charset_seg_count`.
4. Ensure `glyph_table_len == 4 × glyph_count`.
5. Ensure `atlas_offset == glyph_table_offset + glyph_table_len` and that at least 20 bytes remain for the inner header.
6. After reading `width` and `height`, ensure enough bytes remain for the row mask (`ceil(height/8)`) and that `atlas_offset + 20 + ceil(height/8) ≤ total_len`. Then validate that `atlas_len` can contain all present rows: `payload_bytes = present_row_count * ceil(width/2)`. **Producers SHOULD set** `atlas_len = 20 + ceil(height/8) + payload_bytes` for exact sizing.
7. If `kerning_offset != 0`: ensure `kerning_offset ≥ atlas_offset + atlas_len` and space for `7 × kerning_count` exists.
8. Decode only the inner header (7.1) to get `atlas.width/height`, then ensure for each glyph that `x + w ≤ atlas.width`.
9. Metric sanity (recommended but not mandatory): `line_height ≥ ascent - descent`, `ascent ≥ 0`, `descent ≤ 0`.

---

## 10. Example JSON (producer‑side sidecar)

This JSON is **not** part of MFNT; it is used by the reference builder. With charset mode, glyphs omit codepoints entirely.

```json
{
  "line_height": 16,
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
