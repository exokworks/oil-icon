# 上游同步规范

本仓库以 `origin`（个人/团队远端）为**唯一推送目标**；`upstream`（原项目）仅用于拉取。不要运行 `git push upstream`，也不要对本仓库使用不指定远端的 `git push`、`git push --all` 或 `git push --mirror`。本地 `main` 应继续跟踪 `origin/main`，不要改为跟踪 `upstream/main`。

## 远端确认

```bash
git remote -v
# 如果没有 upstream：
git remote add upstream https://github.com/oil-oil/oil-icon.git
```

确认 `origin` 指向自己的远端；如果已有 `upstream` 但地址不同，先核对再调整，勿覆盖未知配置。

## 从上游同步到本地

1. 运行 `git status --short --branch`。若有未提交的工作，先提交到合适的分支，或暂存并在同步后恢复；不要直接覆盖工作区。
2. 获取上游：`git fetch upstream main`。这只更新远端跟踪引用，不会改动工作区或推送代码。
3. 检查分叉和上游变更：

   ```bash
   git log --oneline --left-right main...upstream/main
   git diff --stat main..upstream/main
   ```

4. 确认工作区干净，再执行 `git switch main` 和 `git merge --ff-only upstream/main`。没有新提交时无需合并；若快进失败，说明本地也有独立提交，先检查分叉，在本地明确执行 `git merge upstream/main` 并处理冲突，不要强推或重写上游历史。
5. 检查受影响的功能、测试和文件；如有本地差异，更新[本地更新简史](LOCAL_CHANGES.md)，确认后仅执行 `git push origin main`。不要推送到 `upstream`。

上游同步不会自动完成本地未提交改动、冲突处理或推送。不要把 `git pull upstream main` 作为默认操作：先 `fetch` 再审阅并合并，便于发现上游与本地差异。
