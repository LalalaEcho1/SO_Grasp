# SO_Grasp 项目 AI 协作说明

硕士论文项目：基于阻碍程度（OD）扩展关系图与自适应决策的杂乱堆叠场景顺序抓取。
本文件是所有 AI（云端 Claude / 服务器 Claude Code / Codex）接手时的第一份必读文档，内容与 `AGENTS.md` 相同。

## 接手原则（必须遵守）

- 优先读现有代码和测试，不要凭印象重写；未跟踪文件不等于垃圾文件，不许删除或回滚。
- 路径里有空格（`D:\Stacked-Object Grasping`、WSL `/mnt/d/Stacked-Object Grasping`），命令一律加引号。
- 缺依赖或缺数据导致测试失败时，先跑小测试定位缺什么，不要改逻辑绕过。
- 论文主线低风险优先：关系图、自适应评分、对比实验和结果表稳定之前，不动完整 RL。
- OD 是边特征之一（edge(A→B) = A 对抓取 B 的阻碍），不要写成"遮挡+支撑+高度+接触之和"。

## 分工协议

- **云端 Claude（Cowork）**：代码审查与设计、新功能与单测、实验结果分析、论文表格与写作。无 mujoco、连不上 GitHub 与服务器。
- **服务器 Claude Code / Codex**：一切要碰真实环境的事——mujoco 动态验证、全量实验、GraspNet 大数据、渲染、git push。
- **用户**：决策（入库、论文表述、参数取舍）、提供数据路径、审核改动。
- git 仓库是唯一事实源；每次 AI 会话结束在 `docs/ai_session_log.md` 追加会话记录（日期、做了什么、仓库状态、留给下一个 AI 的事）。
- 同一时间一个模块只有一个 AI 在改；要并行就开分支。

## 环境与命令速查

```bash
# WSL / Linux 环境
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests        # 全量测试（无 mujoco 时约 7 个环境项失败，属正常）
python scripts/run_scene.py --headless --json                    # 关系图 smoke
python scripts/run_formal_main_v1_experiment.py --limit-scenes 3 --max-steps 1 --no-save   # formal smoke
```

GraspNet 数据布局：`<prediction-root>/scene_XXXX/realsense/帧号.npy`；场景数据 `data/graspnet/scenes/scene_XXXX/realsense/{rgb,depth,label,annotations,camK.npy,camera_poses.npy,cam0_wrt_table.npy}`。split 配置 `configs/graspnet_candidate_split.json`（scene_level，禁止随机拆分同场景帧）。

## 当前状态与已定结论（2026-07-26）

- `adaptive_score_v2` 排序键（低风险→高度→抓取风险→综合分→名称）是**刻意设计**，有单测保障；不要"修复"。
- 真实数据管线（scene_0007，40 帧，首抓）：基线 57.5% 成功的瓶颈在**候选供给**，不在排序。已定推荐参数：`--clamp-width --collision-threshold 0.02`（绑定参数保持默认 r=3, tol=0.12）→ 87.5%，risk 阈值 0.45 不动。
- 绑定诊断结论：放宽深度容差/半径只提高绑定率不提高成功率（视差导致像素落在后方表面，深度检查在正确拦截绑错）；30% 候选打在背景上属上游预测质量问题。下一步方法改进：**3D 最近物体绑定**。
- 供给修复后多策略首抓对比（scene_0007）：adaptive-score-v2-graspnet 85% >> od-only 60% > lowest-blocked 57.5% >> highest-first 35% ≈ random 32.5%。真实数据上可行性门控是决定性差异（抽象实验中 highest-first 曾与 v2 打平）。
- 待办：结论需在 split_v1 五场景（0007/0009/0011/0015/0017）上验证泛化；risk 阈值需用 Robotiq 动态验证标定（`scripts/calibrate_risk_threshold.py` 已就绪）；v3（adaptive-score-v3-candidate）代码在服务器，需同步进主仓库。

## 关键入口

- 关系图/OD：`src/stacked_grasping/relations/{graph,obstruction_degree,geometry}.py`
- 决策：`src/stacked_grasping/planning/{adaptive_score_v2,policies,episode,grasp_risk}.py`
- 点云可行性与绑定：`src/stacked_grasping/gripper/{pointcloud_feasibility,graspnet_binding}.py`
- 诊断工具：`scripts/{diagnose_graspnet_binding,sweep_pointcloud_feasibility_grid,calibrate_risk_threshold}.py`
- 实验入口：`scripts/{run_formal_main_v1_experiment,run_graspnet_split_policy_comparison,run_external_graspnet_pointcloud_episode}.py`
