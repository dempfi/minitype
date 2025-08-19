![minitype](./docs/minitype.png)

MiniType is a tiny, fast bitmap font format designed for embedded devices and anyone passionate about maximizing performance on minimal hardware 💾🚀

In many microcontroller projects, pixel fonts are the go-to choice — they're simple and efficient but often look rough, emphasizing the constraints of the device. MiniType is combining the compactness and speed of pixel fonts with high-quality antialiased text rendering. It supports alpha blending and multiple sizes, delivering crisp, modern typography without compromising performance or memory.

By pre-blending antialiased glyphs with hinting for sharpness and compressing the font data, MiniType enables lightning-fast O(1) glyph lookup and rendering. The result is smooth, readable text that's both visually appealing and efficient—a perfect balance of simplicity and quality.

**Key benefits:**

- 💾 _Extremely compact_ — A full ASCII font at 13px occupies just about 4kb
- 🚀 _Instant glyph lookup_ — Direct indexing ensures consistent, fast glyph access even on slow MCUs
- ✨ _Smooth alpha blended edges_ — Precomputed antialiasing makes text look polished and easy on the eyes
- 🔧 _Developer-friendly format_ — Simple to parse, straightforward to generate, with no runtime surprises
- _Effortless conversion_ — Use the provided CLI tool to convert any OTF/TTF font into MiniType

## Screenshots & Demos

- See how MiniType stacks up against TTF on small displays—smaller font sizes without sacrificing clarity.
- Experience smooth, jag-free tiny glyphs with alpha blending that bring pixel fonts to life.
- Explore flash usage charts that reveal exactly how much space you save—ideal for tight embedded environments.
- Discover how MiniType handles multiple font sizes like 8px, 12px, and 16px seamlessly, without extra RAM or CPU overhead.
- Watch glyphs decode and render on the fly in the streaming animation demo—no buffering, no delays.

Imagine your project running faster, looking better, and packing more features because your fonts finally fit just right. Dive into the demos or plug MiniType into your next embedded project and see the difference! 🚀✨

## How to use?

1. Generate your bitmap font

```sh
cargo install minitype
minitype --ttf ./SF-Compact-Rounded-Medium.otf --size 13 -o ./assets/sf_13.mtf
```

2. Embed in your project

```rs
use minitype::{MiniTypeFont, MiniTextStyle};
use embedded_rgba::*;

const SF_13: MiniTypeFont = MiniTypeFont::raw(include_bytes!("./assets/sf_13.mtf"));
let text_style = MiniTextStyle::new(&SF_13, Rgb565::WHITE);

Text::with_alignment("Hello, world!", Point::new(10, 10), text_style, Alignment::Left)
  .draw(&mut canvas.alpha())
  .unwrap();
```
