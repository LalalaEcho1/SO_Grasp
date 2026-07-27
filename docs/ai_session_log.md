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
6. （同日晚间追加）实现 **3D 最近物体绑定** `bind_graspnet_records_to_objects_3d`（graspnet_binding.py，含 5 个单测；`bind_graspnet_records` 统一分发，脚本加 `--binding-mode {pixel,3d}` 与 `--binding-3d-max-distance-m`）。scene_0007 端到端：`3d, maxdist=0.03` + clamp + ct0.02 → 绑定 54.1%、可行 **9.7/帧**（pixel 为 6.9）、成功率 87.5% 持平且不再有视差绑错；maxdist=0.05 过松（82.5%）。**新推荐操作点：`--binding-mode 3d --binding-3d-max-distance-m 0.03 --clamp-width --collision-threshold 0.02`**，待 split_v1 验证。

7. （同日深夜追加）失败帧解剖 + **adaptive-score-v2-riskaware** 新策略。oracle 分析显示 5 个失败帧全部是选择失误（每帧都存在完整风险<0.45 的可行物体），病因：v2-graspnet 排序按夹爪碰撞风险字典序排列，但成败判定用的是含夹爪项的完整风险——选择与裁判度量不一致。新策略用裁判同款完整风险做门控（预测成功集合内仍保持"高度优先"的 v2 精神；全部预测失败时回退到最低完整风险），作为独立策略加入 VALID_POLICIES，不改动任何既有策略；`select_object`/episode 增加 risk_config 传递。scene_0007 40 帧、3d 绑定 0.03 + clamp + ct0.02：**首抓成功率 100%（40/40）**，可行候选 9.7/帧。全链路：57.5% → 87.5%（供给修复）→ 100%（选择对齐），risk 阈值 0.45 全程未动。

**仓库状态**：main @ d2d9dcc + 本次未提交改动（riskaware 策略 3 个代码/测试文件 + 文档更新）。`external/cloud_transfer/` 为传输中转（gitignored），可清理。

8. （继续追加）**标签感知多步清场**：episode 脚本支持 `--max-steps`（<=0 清到失败/清空）与 `--policy`；已抓物体的点按标签逐步从碰撞点云中移除、每步重算可行性（新单测验证"清掉遮挡物后目标解锁"；基线单步结果逐位复现不受影响）。scene_0007 多步清场（3d0.03+clamp+ct0.02）：riskaware 清空率 **44.6%**（均 4.7 步）> graspnet 33.4% > od-only 16.9% > highest-first 13.8% > random 4.2%——策略优势在多步下继续放大。**框架性限制**：静态单帧意味着候选集固定，不能模拟真实系统"每抓一个重新感知/重新预测"，后段物体候选枯竭导致清空率天花板偏低（0 帧全清，尾部失败一半是 gripper-infeasible）——论文中多步数字应作为相对比较与下界呈现，或由 MuJoCo formal 实验承担完整清场指标。

**重要注意**：以上均为 scene_0007 单场景、risk-threshold 判定下的结果——泛化关口：split_v1 五场景验证、Robotiq 动态验证标定（模型内成功≠物理成功）、以及多步清场的静态帧限制（见第 8 条）。论文表述在验证完成前不要引用 100%/44.6% 这些数。

**服务器执行手册（无 Claude Code 版，Codex 或用户手动均可）**

前提：本仓库已 `git pull` 到最新 main。若发现本机有 adaptive-score-v3-candidate 代码，先合并进仓库、跑全量测试、commit + push——这是第一优先级。

split_v1 验证（在服务器 SO_Grasp 目录执行）：

```bash
source .venv/bin/activate  # 或本机等效环境
python -m unittest discover -s tests   # 预期全绿（本机有 mujoco）

# 1) 主表：15 个 selected 帧 × 全策略 × pixel/3d 两种绑定
for m in pixel 3d; do
  python scripts/run_graspnet_split_policy_comparison.py \
    --config configs/graspnet_candidate_split.json \
    --prediction-root data/graspnet_predictions/senior_improved_pointnetplus_split_v1 \
    --policies adaptive-score-v2-riskaware adaptive-score-v2-graspnet od-only highest-first lowest-blocked random \
    --clamp-width --collision-threshold 0.02 \
    --binding-mode $m --binding-3d-max-distance-m 0.03 \
    --out-dir results/split_v1_policy_comparison_$m
done

# 2) 逐场景漏斗与绑定诊断（验证 scene_0007 结论泛化）
for s in 0007 0009 0011 0015 0017; do
  python scripts/sweep_pointcloud_feasibility_grid.py \
    --realsense data/graspnet/scenes/scene_$s/realsense \
    --prediction data/graspnet_predictions/senior_improved_pointnetplus_split_v1/scene_$s/realsense \
    --binding-mode 3d --binding-3d-max-distance-m 0.03 \
    --out-dir results/split_v1_grid/scene_$s
  python scripts/diagnose_graspnet_binding.py \
    --realsense data/graspnet/scenes/scene_$s/realsense \
    --prediction data/graspnet_predictions/senior_improved_pointnetplus_split_v1/scene_$s/realsense \
    --out-dir results/split_v1_binding/scene_$s
done
```

第一个看的数：主表的 `missing_prediction_count`（应为 0）；然后各场景 feasible/帧 与成功率是否复现 scene_0007 的量级。

**实验记录规范（重要）**：`results/` 在 .gitignore 里，原始输出留在跑实验的机器上即可；每个实验完成后，把小体积汇总文件（`summary.json`、`*_summary.csv`、`grid_summary.*`、`binding_diagnosis.json`、`binding_sweep.csv` 等）拷贝到被 git 跟踪的 **`docs/experiment_records/<日期>_<主题>/<实验名>/`**，连同 `docs/ai_session_log.md` 的会话条目一起 commit + push。这样实验记录随仓库同步到每台机器（用户 D 盘 pull 后本地即有），云端 Claude 拉取后出论文表格。首批示例见 `docs/experiment_records/2026-07-26_scene0007_cloud/`。

第三件事（只有本机能做）：Robotiq 动态验证批量跑 split 帧的实际抓取，产出含 `validation.lift_success` 的 summary JSON 目录，之后用 `scripts/calibrate_risk_threshold.py --frame-results <episode frame_results.csv> --validation-dir <目录>` 标定 risk 阈值。

**留给下一个执行者**

- [ ] 用户：`git add -A && git commit && git push`（本机）；删除仓库根 `_to_delete/`。
- [ ] 服务器（Claude Code / Codex）：同步 v3（adaptive-score-v3-candidate）进仓库；在 split_v1 五场景上跑 `diagnose_graspnet_binding` 与 `sweep_pointcloud_feasibility_grid` 验证推荐参数泛化；跑 Robotiq 动态验证批量，产出 (risk, lift_success) 对，供 `calibrate_risk_threshold.py` 标定阈值。
- [ ] 云端 Claude：拿到 split_v1 结果后出论文表格（按 difficulty/场景分组 + 配对检验）。3D 绑定已完成，服务器验证时用 `--binding-mode 3d --binding-3d-max-distance-m 0.03` 与 pixel 模式各跑一遍以便对比。
