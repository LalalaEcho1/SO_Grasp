# AI 会话日志

约定：每次 AI 会话结束追加一段（新条目放最上面）：日期、执行者、做了什么、仓库状态、留给下一个执行者的事。

## 2026-07-26 · 云端 Claude（Cowork）

**做了什么**

1. 入库三个未跟踪项并提交 `be0eef6`（assets/scenes 全部场景、handoff 文档、split 对比测试；assets/objects 大网格仍按 .gitignore 排除）。
2. 新增供给侧诊断与修复三件套（含 16 个单测，全量 194 用例通过）：
   - `pointcloud_feasibility.py` 新增 `clamp_width_to_max_opening`（默认关，`--clamp-width` 接入两个运行脚本）
   - `scripts/sweep_pointcloud_feasibility_grid.py`、`scripts/diagnose_graspnet_binding.py`、`scripts/calibrate_risk_threshold.py`
3. 把 scene_0007 的 40 帧必需数据（label/depth/annotations/相机参数）经桥接搬到云端，逐位复现 6 月基线（46.6% 绑定 / 3.2 可行帧 / 57.5% 成功）。
4. 云端实验结论（结果存 `results/cloud_runs/20260726_scene0007/`）：
   - 推荐参数 `--clamp-width --collision-threshold 0.02`（绑定默认 r=3 tol=0.12）：可行候选 3.2→6.9/帧，成功率 57.5%→**87.5%**，risk 阈值 0.45 未动。
   - 放宽绑定（tol 0.2 或 r=5）提高绑定率但不提高成功率；带符号深度偏移分析（mismatch 中位 dz=-0.174m，全为负）证明是视差落到后方表面，深度检查在正确拦截绑错——**不要放宽容差**，改进方向是 3D 最近物体绑定。
   - 30%（1184/4000）候选 15px 内无任何标签，属 GraspNet 预测打在背景，是上游候选质量问题，可作为论文指标。
   - 供给修复后多策略首抓对比：adaptive-score-v2-graspnet **85%** >> od-only 60% > lowest-blocked 57.5% >> highest-first 35% ≈ random 32.5%。可行性门控是真实数据上的决定性差异。
5. 写入 `CLAUDE.md` / `AGENTS.md`（本协作说明）与本日志。

**仓库状态**：main @ be0eef6 + 本次未提交改动（10 个代码/测试文件 + 3 个文档 + results/cloud_runs）。`external/cloud_transfer/` 为传输中转（gitignored），可清理。

**留给下一个执行者**

- [ ] 用户：`git add -A && git commit && git push`（本机）；删除仓库根 `_to_delete/`。
- [ ] 服务器（Claude Code / Codex）：同步 v3（adaptive-score-v3-candidate）进仓库；在 split_v1 五场景上跑 `diagnose_graspnet_binding` 与 `sweep_pointcloud_feasibility_grid` 验证推荐参数泛化；跑 Robotiq 动态验证批量，产出 (risk, lift_success) 对，供 `calibrate_risk_threshold.py` 标定阈值。
- [ ] 云端 Claude：拿到 split_v1 结果后出论文表格（按 difficulty/场景分组 + 配对检验）；实现 3D 最近物体绑定（配单测）。
