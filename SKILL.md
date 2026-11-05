---
name: oil-icon
description: "生成风格统一的图标集合，通过宫格生图、切分和去背景交付透明 PNG。用户需要成套自定义图标、栏目图标或匹配品牌风格的图标时使用；不用于单个简单矢量图标或小尺寸密集功能符号。"
---

# oil-icon

Generate a **set** of icons that share one visual style, output as transparent PNGs.

Image generation runs on a flat grey backdrop as a 4×4 sheet — 16 icons in one generation, 有助于统一风格，但仍需逐个检查 — then each cell is sliced out and background-removed.

Agent owns the design decisions (which style, which palette, which metaphors, whether to match a brand); image generation and slicing are mechanical steps.

## 外部 API 环境变量

内置生图工具不需要额外 Key。使用已授权的 OpenAI Images 外部 API 时，脚本直接读取可信运行环境中的 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`、`OPENAI_USER_AGENT`）；未设置 Key 则停止，告知用户在自己的环境中配置，不在聊天中索取或传递密钥。

## Workflow

1. **Choose the style.** Either pick a built-in from `styles/` (table below), or **match the user's brand** by reading their project's design tokens / assets and deriving a custom style-spec — see `references/style-adaptation.md`. Freeze the result as a style-spec with the same fields as the built-in JSONs; 后续批次复用同一规范，并检查生成偏差.
2. **List the icons.** Write each concept as `name — short concrete metaphor` (e.g. `settlement — a wallet with a coin`). One clear idea per icon; keep the detail level consistent across the set. 默认每张 **16** 格；明确选择 **9** 格时，提示词使用 3×3，切图命令必须同时传 `--grid 3`。 If the user asks for an exact count below 16, still use a **4x4 / 16-cell sheet**: place the requested icons first in row-major order, then explicitly ask the remaining cells to stay empty grey background.
3. **Compose the prompt.** `style.preamble` + the style's `construction` fields (as short guidance) + palette-lock line + numbered metaphors + the fixed composition text — see `references/prompt-template.md` and `references/construction.md`.
4. **Generate the sheet(s).** Choose an image provider by capability (see `references/image-providers.md`):
   1. **Built-in / host imagegen first** — if the running agent has a callable image-generation tool, including Codex's built-in `image_gen`, the `imagegen` skill, or an MCP, call it directly.
   2. **External API otherwise** — if no callable image-generation tool is available, use the supplier the user has authorized; ask for a supplier if none is selected. For OpenAI Images, `scripts/gen_image.py --prompt "…" --out raw/<style>.png` reads `OPENAI_API_KEY` directly from the environment. If it is missing, stop and ask the user to configure it in their trusted runtime; never request a Key in chat.
   Save or copy each generated sheet to `raw/<style>.png` on flat grey `#808080`. Built-in imagegen may save under `$CODEX_HOME/generated_images/...`; after generation, move or copy the selected PNG into the task output `raw/` folder. 同一张图更容易保持共同风格，但仍须逐个验收，不能把一次生成视为一致性保证； a provider only has to turn the prompt into a grey-background PNG.
5. **Prepare & slice.** Set `SKILL_DIR` to the absolute directory containing this `SKILL.md`; keep generated files in the task directory. Run the installer before first use (or to repair/recheck the environment):
   ```bash
   bash "$SKILL_DIR/scripts/setup.sh" && \
   "$SKILL_DIR/.venv/bin/python" "$SKILL_DIR/scripts/slice_icons.py" "/absolute/task/raw/style.png" "/absolute/task/out" --mode rembg
   ```
   Setup installs **all** dependencies into the Skill's `.venv`, prepares U²-Net and verifies CPU inference. It selects Python 3.12/3.13 (3.12 recommended); if needed, pass `--python /path/to/python3.12`. An incompatible existing `.venv` must be moved aside before rebuilding; never clear it automatically. Only proceed after setup reports ready. Use the style's `cutout` mode (`floodfill` flat/hard-edge, `rembg` soft / 3D / glossy / photo), with `--thresh N`, `--grid 3|4` and `--count N` as needed. Setup and slicing share `.venv/models` automatically; respect existing `U2NET_HOME` / `REMBG_HOME` overrides. See [installation details](README.md#配置依赖与使用边界) for network and recovery requirements.
6. **QA (mandatory).** Composite the sliced PNGs onto a contrasting colour (e.g. magenta) and check every icon for (a) a fragment bled in from a neighbouring cell, (b) leftover grey, and (c) consistency against the style's `construction` — uniform stroke, one radius, matched detail level, shared parts, motif present, single perspective, and no accent out-weighting the defining element (in line styles the accent must be lighter and smaller than the stroke). Re-slice or regenerate any that fail — never ship an icon carrying part of the one above it.
7. **Deliver** the transparent PNGs, organized and named.

## Built-in styles

| style | look | cutout | fits |
|---|---|---|---|
| linear | mono line / outline | floodfill | utility, settings |
| filled | solid single-colour glyphs | floodfill | mobile nav / tab bars |
| colorblock | bold flat multi-colour blocks | floodfill | kids, creative |
| cartoon | sticker-style cartoon | rembg | games, entertainment |
| isometric | isometric 3D miniatures | rembg | SaaS, dashboards |
| render3d | glossy puffy 3D | rembg | rewards, onboarding |
| sticker | die-cut glossy vinyl | rembg | social, reactions |
| realistic | photoreal product renders | rembg | e-commerce, hero |
| animal-badge | animal emblems in badges | rembg | avatars, kids |

Each `styles/<name>.json` holds the preamble, locked palette, cutout mode, threshold, construction system, and what it fits.

## Key rules (see `references/design-rules.md`)

- **Lock the palette as hex** in the spec — colourful styles drift run-to-run otherwise.
- **Never make black filled icons** — near-black / black may be used as a thin outline or line stroke, but never as the dominant filled mass of an icon set. If the brand foreground is black, map it to strokes only; for filled or duotone styles, choose a mid-light brand colour for the main shape and keep dark colours small.
- **Start from the real product context** — when a user gives a live site, screenshot, mascot, logo, or brand asset, derive the icon concepts and construction from that source first. Do not jump straight into generic style families such as isometric, duotone, 3D, or colorblock; those are finishes applied after the product motif, real navigation, and real feature names are locked.
- **Slicing drops neighbour bleed** — the slicer keeps only the centred icon per cell.
- **Cutout by edge type** — flat/hard-edge → floodfill; soft / 3D / glossy / photo → rembg.
- **Design sophistication comes from a construction system + a brand motif**, not from tinting a generic set — each style's `construction` object encodes the grid, stroke, radius, detail budget, shared parts, and motif (see `references/construction.md`). A rigorous system ultimately wants vector; raster is for direction and spot icons.
