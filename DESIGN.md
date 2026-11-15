---
name: DubBench
description: Dark-default Avalonia theme for the Trackdub benchmark harness. Not the commercial desktop shell.
colors:
  accent: "#00BFA5"
  accent-light: "#009688"
  ground: "#1A1A1A"
  ground-light: "#F5F5F5"
  sidebar: "#1E1E1E"
  sidebar-light: "#EEEEEE"
  content: "#252525"
  content-light: "#FAFAFA"
  surface: "#2D2D2D"
  surface-light: "#FFFFFF"
  text: "#E8E8E8"
  text-light: "#212121"
  text-secondary: "#9E9E9E"
  text-secondary-light: "#757575"
  border: "#3D3D3D"
  border-light: "#E0E0E0"
  tab-inactive: "#6B6B6B"
  tab-inactive-light: "#9E9E9E"
  error: "#FF5252"
  error-light: "#D32F2F"
  success: "#4CAF50"
  success-light: "#388E3C"
  warning: "#FF9800"
  warning-light: "#F57C00"
typography:
  title:
    fontSize: "20px"
  heading:
    fontSize: "18px"
  body:
    fontSize: "14px"
  caption:
    fontSize: "12px"
  mono:
    fontSize: "13px"
  micro:
    fontSize: "10px"
rounded:
  row: "4px"
  panel: "6px"
spacing:
  rail: "4px"
  field: "4px"
  stack: "12px"
  panel: "12px"
  content: "16px"
components:
  nav-button:
    backgroundColor: "transparent"
    textColor: "{colors.tab-inactive}"
    padding: "0"
    size: "18px"
    height: "36px"
    width: "36px"
  result-panel:
    rounded: "{rounded.panel}"
    padding: "12px"
  run-button:
    height: "36px"
    width: "200px"
  leaderboard-row:
    backgroundColor: "{colors.sidebar}"
    rounded: "{rounded.row}"
    padding: "10px"
---

# Design System: DubBench

## Overview

**Creative North Star:** not chosen. The qualitative round was unanswered. Do not invent a metaphor, a poetic color name, or a component mood for this file.

This file records the only visual system in the Trackdub public core: DubBench, the Avalonia benchmark harness under `src/DubBench`. It is not the commercial workstation shell. That shell lives in the private desktop repo and was not scanned.

DubBench requests the dark theme (`RequestedThemeVariant="Dark"` in `App.axaml`) and loads Avalonia `FluentTheme` for control chrome. Color and type size live in two resource dictionaries with the same keys: `src/DubBench/Resources/DubBenchTheme.axaml` and `DubBenchThemeLight.axaml`. The frontmatter uses the dark values. Keys ending in `-light` are the light dictionary, not a second opinion about the dark token.

The implemented shell is a 50px icon rail plus a content pane. Depth is a stack of grays. Several brushes are defined and never referenced by a view or by C#.

**Key Characteristics:**

- Dark is the default. Light is a paired dictionary, not a separate product.
- FluentTheme supplies font, hover, and focus. DubBench sets none of those.
- Depth is tonal. No shadow is defined.
- The accent, status, and primary-button brushes are defined and unwired.
- Corners in views are 4px or 6px, as literals, not as theme resources.

## Colors

Dark is the normative theme. Light values are the same resource key in the other dictionary. Names below are the resource keys. Poetic names were not chosen.

`TabActiveBrush` is the same hex as `ThemeAccentBrush`. `ThemeBorderBrush` is the same hex as `BorderBrush`. In the dark theme, `ButtonPrimaryText` is the same hex as `ThemeBackgroundBrush`. Those are aliases, not extra pigments.

### Primary

- **ThemeAccentBrush** (`#00BFA5`, light `#009688`): defined as the accent, and copied into `TabActiveBrush` and `ButtonPrimaryBackground`. No view and no C# file under `src/DubBench` references `ThemeAccentBrush`, `TabActiveBrush`, or `ButtonPrimaryBackground`.

### Neutral

- **ThemeBackgroundBrush** (`#1A1A1A`, light `#F5F5F5`): window background. Applied on `BenchmarkWindow`.
- **SidebarBackgroundBrush** (`#1E1E1E`, light `#EEEEEE`): the icon rail, and the leaderboard row fill.
- **ContentBackgroundBrush** (`#252525`, light `#FAFAFA`): the content pane. Applied on `BenchmarkWindow`.
- **ThemeSurfaceBrush** (`#2D2D2D`, light `#FFFFFF`): defined. Not referenced by the views scanned.
- **ThemeForegroundBrush** (`#E8E8E8`, light `#212121`): text on headers, status, and subtitles. `ThemeTextPrimaryBrush` is the same dark hex and is not referenced.
- **ThemeTextSecondaryBrush** (`#9E9E9E`, light `#757575`): defined. Not referenced. Subtitles use `ThemeForegroundBrush` at opacity 0.7 instead.
- **BorderBrush** (`#3D3D3D`, light `#E0E0E0`): separators and 1px panel strokes.
- **TabInactiveBrush** (`#6B6B6B`, light `#9E9E9E`): icon-rail button foreground. Applied.

### Named Rules

**The Unwired Brush Rule.** A brush in the theme dictionary is not a component style until a view or code-behind references it. Accent, tab-active, primary-button, success, warning, and error are defined and unused in the views and C# scanned for this file.

**The Alias Rule.** Do not add a new pigment for `TabActiveBrush`, `ThemeBorderBrush`, or dark `ButtonPrimaryText`. They repeat an existing hex.

## Typography

**Display Font:** none set.
**Body Font:** none set. `App.axaml` loads `FluentTheme` and no `FontFamily`.
**Label/Mono Font:** `MonospaceFontSize` is defined at 13. No view sets a monospace family.

**Character:** not chosen. The dictionaries store sizes only. Views do not bind those size keys.

### Hierarchy

Sizes are Avalonia device-independent pixels. The frontmatter writes them as `px` so the file stays portable. That unit label is a translation, not a second source.

- **Title** (20): `FontSizeTitle`. Defined. Not referenced.
- **Heading** (18): `HeadingFontSize`. Not referenced. Page titles hardcode `FontSize="18"` and `FontWeight="SemiBold"`.
- **Body** (14): `BodyFontSize`. Defined. Not referenced.
- **Label** (12): `CaptionFontSize`. Defined. Not referenced. One status path hardcodes `FontSize="11"`, which is not in the scale.
- **Mono** (13): `MonospaceFontSize`. Defined. Not referenced.
- **Micro** (10): `FontSizeMicro`. Defined. Not referenced.

### Named Rules

**The Hardcoded Size Rule.** Do not treat the typography keys as live. Views hardcode 18, and one view uses 11, which the dictionary does not define.

## Layout

The window is 960 by 720, minimum 640 by 480, centered (`BenchmarkWindow.axaml`). The grid is `50,*`: a fixed icon rail and a stretching content pane. Content margin is 16. Tab stacks use spacing 12. Form groups and the rail use spacing 4. Combo boxes that set a width use 280. The run-count control uses 120. These are view literals, not spacing resources. The frontmatter `spacing` keys name those repeated literals so later edits can point at them. They are not in the theme dictionary.

## Elevation & Depth

No box shadow, drop shadow, or elevation resource exists under `src/DubBench`. Depth is the dark tone stack: ground `#1A1A1A`, rail `#1E1E1E`, content `#252525`, surface `#2D2D2D`. The light pair is the reverse idea: paper `#F5F5F5`, rail `#EEEEEE`, content `#FAFAFA`, surface `#FFFFFF`. Hover and focus elevation, if any, come from `FluentTheme` and are not tokenized here.

### Named Rules

**The Tone Stack Rule.** Do not add a shadow to separate rail, content, and panels. The dictionaries already separate them by tone, and the views use a 1px border where a region needs an edge.

## Shapes

Views use two corner literals. Result panels use `CornerRadius="6"`. Leaderboard rows and the nested export-folder border use `CornerRadius="4"`. Neither value is a theme resource. The frontmatter `rounded` keys name those literals. Rail buttons are 36 by 36 with no local corner radius, so their shape is whatever `FluentTheme` does with a transparent button. Panel strokes are 1px `BorderBrush`. Icons in the rail are emoji strings in `BenchmarkIcons.axaml`, not drawn geometry.

## Components

Qualitative feel was not collected. The notes below are what the views set, plus what `FluentTheme` is left to supply.

### Buttons

- **Shape:** the rail button sets no radius. The run button sets no radius. Both leave shape to `FluentTheme`.
- **Primary:** `ButtonPrimaryBackground` and `ButtonPrimaryText` are defined and not applied. The run button is a default `Button` at 200 by 36, content "Run Benchmark", left aligned. A secondary action, "Optimize Now", is 140 by 32. "Refresh" is 100 by 28. "Browse..." sets no size.
- **Hover / Focus:** not set in DubBench. `FluentTheme` supplies both.
- **Rail:** transparent background, `TabInactiveBrush` foreground, 36 by 36, padding 0, font size 18, emoji content, tooltip plus `AutomationProperties.Name`.

### Cards / Containers

- **Corner Style:** 6px on result panels. 4px on leaderboard rows.
- **Background:** result panels set no fill, only a 1px border. Leaderboard rows fill with `SidebarBackgroundBrush`.
- **Shadow Strategy:** none. See Elevation & Depth.
- **Border:** 1px `BorderBrush` on result panels and the nested path border. Leaderboard rows have no border.
- **Internal Padding:** 12px on result panels. 10px on leaderboard rows. 8px on the nested path border.

### Inputs / Fields

- **Style:** `TextBox`, `ComboBox`, and `NumericUpDown` are stock `FluentTheme` controls. DubBench sets no stroke, fill, or radius on them.
- **Focus:** not set in DubBench.
- **Error / Disabled:** no field error style is defined. Run buttons bind `IsEnabled` to the inverse of an in-progress flag. Disabled chrome is `FluentTheme`.

### Navigation

A vertical rail in a 50px column. Four buttons, spacing 4, top margin 8. Labels are tooltips: ONNX Model, Audio-Prep, Dubbing, Local Runs. The view sets every button to `TabInactiveBrush`. Nothing in the scanned views or C# swaps in `TabActiveBrush` for the selected tab.

## Do's and Don'ts

Grounded in this checkout only. Not a brief for the commercial shell.

### Do:

- **Do** treat dark as the default, because `App.axaml` sets `RequestedThemeVariant="Dark"`.
- **Do** keep the light dictionary as the same keys with different hex values.
- **Do** separate regions with the tone stack and a 1px border, because that is the implemented depth model.
- **Do** keep rail buttons at 36px inside the 50px column, with a tooltip and an accessible name.

### Don't:

- **Don't** claim the accent or the status brushes are on screen. They are defined and unreferenced.
- **Don't** add a shadow token. None exists.
- **Don't** set a font family in new DubBench views unless the theme dictionary gains one. None is set today.
- **Don't** document the commercial Avalonia shell from this file. It is not in this repo.
- **Don't** invent a creative north star, a poetic color name, or a component mood. That round was unanswered.
