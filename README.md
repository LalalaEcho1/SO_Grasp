# Stacked-Object Grasping

硕士论文原型项目：基于 OD 扩展关系图与自适应强化学习的堆叠物体顺序抓取方法研究。

当前第一版目标不是直接训练强化学习，而是先跑通 MuJoCo 中的堆叠场景、物体状态读取、接触关系读取和初步关系图构建。

## 当前原型包含

- MuJoCo 堆叠物体场景：
  - `assets/scenes/stacked_blocks.xml`
  - `assets/scenes/ycb_lite_stacked.xml`
  - `assets/scenes/ycb_mesh_stacked.xml`，下载官方 YCB mesh 后生成
- 物体位姿与接触对读取
- 初步有向关系图：
  - contact
  - support
  - height_diff
  - xy_overlap_ratio
  - relative_xy_distance
  - OD：基于 top/side 抓取候选与 approach corridor 的抓取路径阻碍估计
- 自适应启发式评分 baseline

## 推荐运行环境

建议在 WSL Ubuntu 里运行，Python 推荐 3.10+。

```bash
cd "/mnt/d/Stacked-Object Grasping"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## 运行

只输出关系图和评分：

```bash
python scripts/run_scene.py --headless
```

运行 YCB-Lite 场景：

```bash
python scripts/run_scene.py --scene assets/scenes/ycb_lite_stacked.xml --headless
```

打开 MuJoCo viewer：

```bash
python scripts/run_scene.py --viewer
```

输出 JSON，方便后续写实验记录：

```bash
python scripts/run_scene.py --headless --json
```

不依赖 viewer，导出场景截图和关系图：

```bash
python scripts/export_visuals.py
```

导出 YCB-Lite 场景图片和关系图：

```bash
python scripts/export_visuals.py --scene assets/scenes/ycb_lite_stacked.xml --gl-backend osmesa
```

下载官方 YCB Google 16k visual mesh，并生成更真实的 MuJoCo visual 场景：

```bash
python scripts/download_ycb_meshes.py --objects starter
python scripts/run_scene.py --scene assets/scenes/ycb_mesh_stacked.xml --headless
python scripts/export_visuals.py --scene assets/scenes/ycb_mesh_stacked.xml --gl-backend osmesa
```

这个版本中，官方 YCB mesh 作为 visual 外观，碰撞和 OD 使用由 mesh 尺寸对齐出来的 box / cylinder 近似。因此截图更真实，关系图、支撑关系和顺序评分也能保持稳定。

如果 `egl` 后端不可用，可以安装 CPU 离屏渲染库后切换到 `osmesa`：

```bash
sudo apt install -y libosmesa6
python scripts/export_visuals.py --gl-backend osmesa
```

导出的文件会放在：

```text
results/visual_debug/
```

## 顺序抓取 episode

抽象顺序抓取实验会复用当前关系图和自适应评分：每一步选择评分最高的物体，模拟抓取成功后把它移出工作空间，再让场景重新稳定并计算下一步。

```bash
python scripts/run_episode.py --scene assets/scenes/ycb_mesh_stacked.xml
```

输出包括：

```text
完整抓取顺序
每一步选择时的 ranking
每一步选择时的关系图 edges
```

JSON 会保存到：

```text
results/episodes/
```

这个 episode 接口是后续强化学习环境的雏形：之后可以把 `adaptive_baseline` 替换成 RL policy，把“移除物体”替换成真实夹爪控制或 MuJoCo 抓取执行。

## 后续路线

1. 继续增强 OD 计算：
   - 加入 diagonal grasp
   - 加入更接近夹爪形状的 swept volume
   - 从 AABB/cylinder 近似升级到 mesh/ray/SDF 检测
   - 可视化每个 grasp approach corridor，用于解释 OD 数值来源
2. 完善自适应评分：
   - OD 权重
   - 支撑风险权重
   - 接触密集程度权重
   - 抓取失败反馈权重
3. 加入 MuJoCo 抓取执行：
   - 先简化为移除目标物体
   - 再加入夹爪和轨迹控制
4. 加入强化学习：
   - state = 关系图特征
   - action = 选择下一个抓取物体
   - reward = 成功抓取、减少阻碍、少碰撞、少失败
