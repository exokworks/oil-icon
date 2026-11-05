# 本地更新简史

只记录本地相对于上游的关键改动；上游提交历史以 `git log` 为准。新增或同步改动时，在此补充日期、状态和简要内容，不把未提交工作写成已发布版本。

- **2026-11-05（本地更新）**：图像生成脚本新增 `OPENAI_BASE_URL`，默认仍请求 OpenAI 官方 API；显式设置 `User-Agent`，支持 `OPENAI_USER_AGENT` 覆盖；补充本地 HTTP 回归测试及配置说明。新增本地更新简史和[上游同步规范](UPSTREAM_SYNC.md)，创建本地第三方参考项目目录 `reference-projects/` 并加入 Git 忽略。README 的安装入口改为本仓库 `exokworks/oil-icon`。
- **2026-09-10（本地基线 `afb2007`）**：补齐 API Key 配置页面与业务读取，明确功能边界。此提交也在上游 `main`，不算本地独有改动。

`upstream` 是本地 Git 远端配置，不随仓库文件提交；新克隆需按[上游同步规范](UPSTREAM_SYNC.md)配置。同步前如有未提交更改，请先处理。
