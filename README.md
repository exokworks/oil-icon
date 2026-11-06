<p align="center">
  <img src="./assets/readme/readme-hero.png" width="100%" alt="oil-icon：一次生成一整套，让每个图标都像来自同一个设计师。">
</p>

<p align="center">
  <img src="./assets/readme/readme-section-what.svg" width="100%" alt="02 oil-icon 是什么">
</p>

生成风格一致的图标集合，完成宫格切分、背景去除和逐个检查，交付透明 PNG。

上面的两组图标使用了完全相同的 16 个概念：配色、网格、魔法棒、图层、品牌、标签、卡片、分类、空状态、营销、产品、角色、切图、透明背景、质检和导出。改变的是整套的画法，因此一组是柔软 3D，另一组是等轴微缩。

它适合产品功能、分类、营销卡片、空状态和品牌化插图；如果只是制作 16–24px 的按钮小图标，直接使用 Lucide 等矢量图标库会更合适。

<p align="center">
  <img src="./assets/readme/readme-section-consistency.svg" width="100%" alt="03 为什么整套生成更统一">
</p>

让模型一张一张画，后面的图标很容易忘记前面的线宽、圆角、视角和颜色。`oil-icon` 把 16 个图标放在同一次生成里，让模型始终看见整套作品；开始画之前，还会先把共同的规则定清楚。

一套图标会共同遵守：

- **同一套画法**：网格、视角、线宽、圆角和繁简程度一致。
- **同一套颜色**：色值直接锁定，避免每一批越画越艳、越画越偏。
- **同一个记号**：反复出现的小火花、切角或其他母题，让图标拥有自己的身份。
- **同一次生成**：16 个图标共享同一段视觉上下文，比逐张生成更容易保持一致。

所以它不是给通用图标换一套颜色，而是先把整套的画法定清楚，再让模型按照这些规则画完。

<p align="center">
  <img src="./assets/readme/readme-section-workflow.svg" width="100%" alt="04 一张图，得到 16 张透明 PNG">
</p>

```text
图标清单 + 风格或项目素材
          ↓
定好颜色、视角、圆角和专属记号
          ↓
在纯灰背景生成一张 4×4 图标图
          ↓
自动切成 16 张图片并移除背景
          ↓
在高对比底色检查灰边、串格和一致性
          ↓
交付 16 张透明 PNG
```

硬边图标使用颜色识别清除背景，3D、贴纸等软边图标使用背景移除模型。切图程序保留本格的主体和独立小部件，清除明确属于相邻格子的碎片。

<p align="center">
  <img src="./assets/readme/readme-section-styles.svg" width="100%" alt="05 支持哪些风格">
</p>

仓库里现在有 9 种内置预设，每一种都有自己的色彩、构造规则和背景移除方式：

| 预设 | 视觉特点 | 更适合 |
| --- | --- | --- |
| `linear` · 线性 | 单色描边、统一线宽、圆角收笔 | 工具、设置和内容密集的界面 |
| `filled` · 面性单色 | 清楚的实心轮廓，只使用一种主色 | 导航、标签栏和小面积强调 |
| `colorblock` · 撞色 | 2～3 种颜色组成简洁的几何色块 | 创意、教育和轻松活泼的产品 |
| `cartoon` · 卡通 | 粗描边、圆润造型和贴纸感 | 游戏、娱乐和角色化内容 |
| `isometric` · 等轴微缩 | 统一视角的立体小物件，结构清楚 | 工作台、功能介绍和 SaaS 产品 |
| `render3d` · 3D 渲染 | 柔软膨胀的形状、糖果色和细腻材质 | 引导、奖励、会员和营销卡片 |
| `sticker` · 贴纸 | 清楚的外轮廓、亮面材质和独立贴纸造型 | 社区、聊天、表情和活动素材 |
| `realistic` · 真实实物 | 接近产品摄影的真实物体 | 电商、商品分类和首屏展示 |
| `animal-badge` · 动物徽标 | 动物角色与统一徽章结构结合 | 头像、儿童产品和角色系统 |

这 9 种只是随仓库提供的稳定起点，不是 `oil-icon` 能做的全部风格。我们还可以：

- **直接修改预设**：替换配色，调整线宽、圆角、视角、材质和细节多少。
- **匹配现有品牌**：读取网站、Logo、截图和设计变量，把真实的颜色与形状规则整理成一套图标风格。
- **从文字描述开始**：没有现成设计也没关系，可以先用一句话确定方向，再生成第一张风格样板。
- **保存专属规则**：风格确认后，把色板、构造方式和专属记号固定下来，之后继续生成的新图标也会沿用同一套规则。

内置预设解决的是快速开始，品牌适配解决的是让图标真正属于这个产品。

<p align="center">
  <img src="./assets/readme/readme-section-start.svg" width="100%" alt="06 怎么使用">
</p>

**方式一 · 执行命令**

```bash
npx skills add exokworks/oil-icon
```

**方式二 · 直接交给 Agent**

把下面这句话发给任何支持 Skill 的 Agent：

```text
请安装这个 Skill：https://github.com/exokworks/oil-icon
```

安装完成后，可以直接描述图标清单和使用场景：

```text
[$oil-icon] 根据这个产品的页面和配色，生成一套 16 个功能图标。
```

如果没有现成品牌，也可以指定一个方向：

```text
[$oil-icon] 为我的 AI 写作工具生成一套柔软 3D 图标，包含创作、改写、翻译、素材、历史和导出。
```

`oil-icon` 会优先使用 Agent 当前可用的图片生成能力；没有内置图片工具时，使用已授权的外部图片 API。切图与检查流程保持一致。

<details>
<summary><strong>仓库里有什么</strong></summary>

- `SKILL.md`：完整工作流与执行规则
- `styles/`：内置风格和构造规范
- `references/`：提示词、品牌适配、图标设计与图片生成说明
- `scripts/`：环境安装、切图和背景移除工具
- `reference-projects/`：本地存放其他参考项目，已加入 Git 忽略（新克隆可运行 `mkdir -p reference-projects`）
- `LOCAL_CHANGES.md`：本地更新简史
- `UPSTREAM_SYNC.md`：从上游拉取与仅向 `origin` 推送的规范

</details>

## 配置、依赖与使用边界

### 安装与切图

准备 **Python 3.12 或 3.13**（推荐 3.12），将下方路径替换为实际位置。依赖一次装齐，无需选择安装模式或手动激活环境。

```bash
SKILL_DIR="/absolute/path/to/oil-icon"  # SKILL.md 所在目录

bash "$SKILL_DIR/scripts/setup.sh" && \
  "$SKILL_DIR/.venv/bin/python" "$SKILL_DIR/scripts/slice_icons.py" \
  "/absolute/task/raw/icons.png" "/absolute/task/out" --mode rembg
```

Agent 根据图标风格选择切图参数：

| 场景 | 参数 |
| --- | --- |
| 3D、贴纸、照片等软边图标 | `--mode rembg` |
| 扁平、硬边图标 | `--mode floodfill` |
| 16 格（默认）／9 格 | `--grid 4`／`--grid 3` |

首次安装需访问 Python 包索引和 GitHub，下载约 **176 MB** 的 U²-Net 模型。环境与模型分别保存在 Skill 的 `.venv`、`.venv/models`；图片和产物保存在任务目录。

<details>
<summary><strong>安装细节与故障处理</strong></summary>

- **依赖**：统一安装 Pillow、NumPy、SciPy 和 `rembg[cpu]`，版本见 [requirements.txt](requirements.txt)。仅用二进制 wheel，并固定 CPU Runtime、Numba 和 LLVM 依赖，避免意外源码编译。
- **Python**：优先复用有效 `.venv`；新建时依次查找 `python3.12`、`python3.13` 和兼容的 `python3`，不使用 3.14。也可用 `bash "$SKILL_DIR/scripts/setup.sh" --python /path/to/python3.12` 指定解释器。
- **环境修复**：已有 `.venv` 不兼容或失效时，先重命名备份再重跑，脚本不会自动清空目录。安装中断或模型下载失败时，解决报错或网络问题后重跑即可。
- **就绪检查**：安装显示进度与失败阶段，只有通过 `pip check`、依赖导入、模型加载和真实 CPU 推理后才报告就绪；重复运行会补齐并重新验证。
- **已有配置**：保留镜像、代理及 `U2NET_HOME`／`REMBG_HOME` 缓存设置；覆盖 pip 用户级安装选项，拒绝 `target`、`prefix`、`root`，避免依赖装到 `.venv` 外。

</details>

### 检查与交付

切图会检测实际背景色、保留本格内的独立小部件；背景采样不可靠时自动尝试 rembg，并逐格提示复核。rembg 也可能误删主体，因此会保留色彩去背候选供比较。输出目录的 `qa/` 包含：

- **检测报告** `report.json`：提示疑似残灰、透明孔、部件损失和候选差异，并记录每格修整参数。
- **对照与预览**：原格、抠图后、清理后的对照图，以及洋红、浅色、深色底预览；使用 rembg 时另存色彩候选。

Agent 对照原图逐张检查并决定修整方式：灰边可独立调整外部去背容差，封闭残底可局部清除，误删细节可逐格改用色彩去背。无需用户选择参数，具体用法见 [切图修整](references/cutout-repair.md)。自动提示不能区分正常镂空、灰色图案与残底，修整后仍须复核。

**使用边界**：一次生成不保证整套一致性；16–24px 的小尺寸功能图标优先使用矢量库。

## 外部图片 API

优先使用 Agent 内置生图工具；没有内置工具时，使用已授权的外部 API。OpenAI Images 脚本从运行环境读取以下配置，**不要把 Key 发到聊天中**。

| 环境变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | 必需，在可信运行环境中配置 |
| `OPENAI_BASE_URL` | 可选，自定义 API 地址 |
| `OPENAI_USER_AGENT` | 可选，自定义请求标识 |

沿用上一节的 `SKILL_DIR`，运行：

```bash
python3 "$SKILL_DIR/scripts/gen_image.py" \
  --prompt "图标设计要求" \
  --out "/absolute/task/raw/icons.png"
```

- **自动选型**：默认 `gpt-image-2.6-sunburst`，由 Agent 按任务决定，通常无需用户指定。
- **自动回退**：2.6 返回 HTTP `400`、`403`、`404` 或 `422` 时，改用 `gpt-image-2.5-sunburst` 重试一次。
- **停止重试**：遇到 `401`、`429`、超时、服务端错误或异常响应时停止，避免重复生成或计费。

<p align="center">
  <a href="https://github.com/oil-oil/beautify-github-readme"><img src="./assets/readme/made-with-beautify.svg" width="300" alt="README made with beautify-github-readme"></a>
</p>

## License

MIT
