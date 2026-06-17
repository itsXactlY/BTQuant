#pragma once

// ============================================================================
// Embedded font data (JetBrains Mono + FontAwesome 6)
//
// PHASE 2.2 of TASK_ULTIMA_MMT_GENESIS_WIRED.md requires embedding the TTFs
// as `static const unsigned char[]` so the build has no runtime file
// dependency. The intent is documented; real TTF bytes are a follow-up.
//
// To enable: drop the following files into this directory at build time
//   - JetBrainsMono-Regular.ttf  (https://www.jetbrains.com/lp/mono/)
//   - FontAwesome6-Free-Solid-900.ttf (https://fontawesome.com/)
// and regenerate this header with `xxd -i <file>.ttf` plus the *_SIZE
// variable. The build stays self-contained — no FontAwesome/font fallback
// at runtime.
//
// Until then, UnifiedThemeSystem::apply_fonts() calls ImGui's default font
// and emits a single warning at startup. The Deep Void palette, zero-radius
// borders, and docking layout are unaffected — only the glyph atlas.
// ============================================================================

#ifndef BTQ_FONT_DATA_HPP
#define BTQ_FONT_DATA_HPP

#include <cstddef>

// Empty placeholder arrays so symbol references resolve cleanly. The
// header compiles, apply_fonts() detects _AVAILABLE=0, and falls back to
// the ImGui default font without breaking the build.
static const unsigned char JetBrainsMonoTTF[1] = {0};
static constexpr size_t   JetBrainsMonoTTF_SIZE = 0;

static const unsigned char FontAwesome6TTF[1] = {0};
static constexpr size_t   FontAwesome6TTF_SIZE = 0;

#define JETBRAINS_MONO_TTF_AVAILABLE 0
#define FONTAWESOME6_TTF_AVAILABLE   0

#endif // BTQ_FONT_DATA_HPP
