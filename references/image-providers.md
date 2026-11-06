# Image-generation providers

The slicing pipeline is provider-agnostic: it only needs a **PNG sheet requested on flat grey `#808080`** from a text prompt. This is a target colour; slicing measures the actual backdrop and reports unreliable samples. Never assume a specific tool is present — choose the provider by the current agent's capabilities, not by the agent name.

## Detection order

1. **Built-in / host imagegen first.** If the running agent has a callable text-to-image tool — Codex's built-in `image_gen`, the `imagegen` skill, another image-generation tool, or an MCP — use it directly. Built-in imagegen may write to `$CODEX_HOME/generated_images/...` instead of a requested path; after generation, move or copy the selected PNG to `raw/<style>.png`.
2. **External API otherwise.** If the agent has no callable image-generation tool, use the already authorized supplier; ask for a supplier if none is selected. For **OpenAI Images (gpt-image)**, `scripts/gen_image.py --prompt "…" --out raw/<style>.png` reads `OPENAI_API_KEY` directly from the environment. If it is missing, stop and let the user configure their trusted runtime; do not request a Key in chat. `OPENAI_BASE_URL` optionally specifies an OpenAI-compatible API root including `/v1`, and `OPENAI_USER_AGENT` overrides the request identifier. For any other API (Replicate, Fal, Stability, a self-hosted model...), call it directly to produce the PNG.

For this icon-sheet workflow, let the agent choose the model without asking the user to make a routine selection. `gen_image.py` defaults to `gpt-image-2.6-sunburst` for detail and composition; choose `gpt-image-2.6-flare` only when speed is a clear priority. The valid 2.6 API IDs are `gpt-image-2.6-sunburst` and `gpt-image-2.6-flare`; bare `gpt-image-2.6` is not a valid ID. After an explicit HTTP rejection (400, 403, 404, or 422), the script automatically retries once with `gpt-image-2.5-sunburst`. It does not retry timeouts, rate limits (429), authentication failures (401), server errors, or malformed responses because the first generation may have been accepted or the failure will not be fixed by changing models. Use `--model` only when a task has a specific model requirement.

## The contract

Any provider must satisfy one thing: **text prompt → one PNG**, rendered as the 4×4 (or 3×3) sheet on a solid flat grey background, targeting `#808080`, roughly square. Exact colour compliance is not assumed; the same background estimation and QA run regardless of provider.

## Provider notes

- **Reference images** — bundled `gen_image.py` 的当前入口使用文本提示；其他宿主或供应商是否支持参考图，按实际工具能力判断。支持时可传入已确认的图标 sheet 作为风格参考，仍需验收偏差。
- **Output size** — providers differ (gpt-image ≈ 1024²). The slicer reads real dimensions, so exact size does not matter.
- **Output path** — host image tools may not accept an exact destination path. Generate first, then move or copy the chosen PNG into the task directory’s `raw/<style>.png`.
- **Background / alpha** — oil-icon asks the image provider for a flat grey `#808080` sheet, not a transparent PNG. The oil-icon slicer creates transparency later; do not use a provider's transparent-output or chroma-key workflow for this step unless the user is doing a non-oil-icon image task.
- **One sheet, not one icon** — always prefer one sheet of 16 (or 9): 一次生成有助于统一，但仍需逐个检查风格. Fall back to one-icon-per-image only if a provider cannot hold a coherent multi-icon layout.
