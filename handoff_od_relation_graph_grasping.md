# Codex 交接文件：OD 扩展关系图与自适应顺序抓取

生成时间：2026-05-20  
用途：切换到 Linux 系统上的 Codex 后，继续围绕硕士毕业论文方案展开。

## 1. 当前推荐主线

建议论文主线：

> 基于 OD 扩展关系图的自适应顺序抓取方法

更完整的表述：

> 基于阻碍程度扩展关系图与自适应决策的杂乱场景顺序抓取方法研究

核心思想：

```text
物体识别 / 位姿估计
↓
构建有向物体关系图
↓
边特征 = OD + 支撑关系 + 高度差 + 接触关系 + 可达性影响 + 视觉遮挡等
↓
自适应评分函数或轻量强化学习选择抓取顺序
↓
执行抓取
↓
更新场景图并重新决策
```

该方案的优点：

- 比单纯复现 OD 论文更有扩展性。
- 比完整做 MSP-Grasp、SP-POMDP 或复杂 GNN+RL 更容易落地。
- 满足硕士论文所需的工作量和创新性。
- 可以先实现启发式自适应方法，后续再包装或扩展为强化学习版本，风险更低。

## 2. 和 OD 论文的关系

已参考文件：

```text
D:/论文/Big论文/zhao-et-al-2021-planning-for-grasping-cluttered-objects-based-on-obstruction-degree.pdf
```

OD 论文核心：

```text
Obstruction Degree
↓
计算物体间抓取阻碍程度
↓
形成 object obstruction matrix
↓
基于阻碍矩阵规划抓取顺序
```

OD 原文中可以抽象成：

```text
O(Oi, Oj) = Oj 对抓取 Oi 的阻碍程度
```

如果转换成有向图，可以写成：

```text
edge(Oj -> Oi) = OD(Oi, Oj)
```

注意方向：

```text
OD(Oi, Oj) != OD(Oj, Oi)
```

因为 A 阻碍抓 B，不代表 B 同样阻碍抓 A。

## 3. OD 的概念边界

重要结论：

```text
OD 不是 “遮挡程度 + 支撑关系 + 高度差 + 接触关系 + 可达性影响” 的总和。
```

更准确地说：

```text
OD = grasp obstruction degree
   = 某个物体对目标物体抓取接近路径的阻碍程度
```

OD 主要描述：

- 夹爪沿某个 grasp approach direction 接近目标物体时，是否被其他物体挡住。
- 某个障碍物会使目标物体的抓取候选变差多少。
- 某个物体是否阻碍另一个物体被抓取。

OD 可以通过论文中的射线束方法计算：

```text
围绕抓取候选的 approach direction 采样一组 rays
↓
检测这些 rays 与其他物体的距离或命中情况
↓
距离越近，阻碍程度越大
```

但一般实现中不一定必须使用射线，也可以用：

- gripper swept volume 碰撞检测
- mesh / SDF 距离
- 点云 / TSDF / occupancy 近似

对于当前毕业论文目标，建议先用论文式 ray-cone OD 或简化 swept-volume OD，保证容易实现和解释。

## 4. 边特征应该怎么设计

推荐边定义：

```text
edge(A -> B) = A 对“抓取 B”的影响
```

推荐边特征向量：

```text
e_{A->B} = [
  OD(A -> B),
  visual_occlusion(A -> B),
  support(A -> B),
  support(B -> A),
  contact(A, B),
  height_diff(A, B),
  reachability_effect(A -> B),
  relative_distance(A, B),
  overlap_ratio(A, B)
]
```

其中：

```text
OD(A -> B)
```

表示 A 对抓取 B 的夹爪接近路径造成的阻碍。

```text
visual_occlusion(A -> B)
```

表示 A 是否从相机视角遮挡 B。注意它和 OD 不同：

```text
visual occlusion: 看不看得见
OD / grasp obstruction: 夹爪能不能接近
```

```text
support(A -> B)
```

表示 A 是否支撑 B。

```text
height_diff(A, B)
```

可定义为：

```text
z_A - z_B
```

用来表达上下层关系。

```text
contact(A, B)
```

表示两个物体是否接触或距离小于阈值。

```text
reachability_effect(A -> B)
```

表示 A 的存在让 B 的可抓取性下降多少。

一种定义：

```text
reachability_effect(A -> B)
= valid_grasps(B without A) - valid_grasps(B with A)
```

归一化定义：

```text
reachability_effect(A -> B)
= 1 - valid_ratio(B with A)
```

更适合论文表达的特征：

```text
blocked_grasp_ratio(A -> B)
best_grasp_od(A -> B)
od_mean(A -> B)
od_max(A -> B)
```

特别推荐保留：

```text
best_grasp_od
blocked_grasp_ratio
```

因为顺序抓取真正关心的是：

```text
当前目标物体是否至少还有一个可行抓法？
如果先拿走 A，B 的可抓性会不会明显提升？
```

## 5. 和组会文档中几篇文章的关系

已参考文件：

```text
D:/研究生学习/组会汇报/5.25组会.docx
```

文档里包含三类方法：

### 5.1 MSP-Grasp

文档摘要：

- 多尺度感知框架。
- 全局上下文感知模块 GCA。
- 区域碰撞预测模块 RCP。
- 局部抓取评估模块 LGE。
- 在 GraspNet、仿真、真实环境中测试。

对当前论文的价值：

```text
借鉴碰撞风险、局部抓取评价、可达性影响。
```

不建议完整复现，原因：

- 工作量大。
- 训练数据和网络模块复杂。
- 容易变成复现 6-DoF 抓取网络，而不是顺序抓取论文。
- 对硕士毕业目标来说风险偏高。

### 5.2 Stacked Object Classification Network + Grasping Order Planning

文档摘要：

- 点云分割。
- 堆叠物体分类。
- 关系探索网络。
- 输入包含物体法向量、平均高度、重叠度。
- 输出堆叠关系，例如上下支撑、倾斜、无关。
- 根据关系安全得分、类别安全得分、尺寸安全得分排序。

对当前论文的价值最大。

可借鉴：

```text
支撑关系
高度差
重叠度
接触关系
安全得分排序
```

当前方法可以理解为对它的扩展：

```text
原方法：
几何关系 + 安全得分 → 顺序抓取

你的方法：
OD + 几何/物理关系 + 自适应权重 → 顺序抓取
```

### 5.3 SP-POMDP

文档摘要：

- 状态空间包含物体位置、类型识别概率、遮挡关系。
- 动作是抓取物体并移动到指定位置。
- 使用 POMCP / MCTS 进行信念树求解。
- 每次动作后更新状态。

对当前论文的价值：

```text
借鉴“每次抓取后重新更新状态并重新决策”的思想。
```

不建议完整实现 POMDP，原因：

- 状态、动作、观测、信念树定义复杂。
- 调参难。
- 工程实现不够稳。

## 6. 推荐创新点

建议论文主打两个创新点：

### 创新点 1：OD 扩展有向关系图

表述：

> 提出一种融合阻碍程度、支撑关系、接触关系、高度差和可达性影响的有向物体关系图，用于描述杂乱堆叠场景中的物体间抓取影响。

对应实现：

```text
node = object-level features
edge = directed relation features
```

节点特征可以包括：

```text
object class
pose
size
height
graspability
number of valid grasps
```

边特征可以包括：

```text
OD
support
contact
height_diff
overlap_ratio
reachability_effect
visual_occlusion
```

### 创新点 2：自适应顺序抓取决策

表述：

> 提出一种基于场景关系特征的自适应顺序抓取决策方法，根据当前堆叠、遮挡和可达性状态动态调整抓取优先级。

实现上建议先做自适应评分函数：

```text
Score(B) =
  w1 * graspability(B)
- w2 * blocked_by_OD(B)
- w3 * support_risk(B)
- w4 * contact_risk(B)
+ w5 * clearance_gain(B)
```

其中：

```text
blocked_by_OD(B) = sum_A OD(A -> B)
support_risk(B) = 抓 B 是否会导致上方物体不稳定
contact_risk(B) = 抓 B 是否容易碰撞或拖动邻近物体
clearance_gain(B) = 抓走 B 后能减少多少其他物体的阻碍
```

权重自适应规则示例：

```text
堆叠严重 → 支撑关系权重大
遮挡严重 → OD 权重大
抓取失败多 → 可达性影响权重大
物体接触密集 → 接触风险权重大
```

如果后续需要强化学习，可以扩展为：

```text
state = 当前关系图特征
action = 选择下一个要抓取的物体
reward = 抓取成功 + 清空率提升 - 碰撞 - 抓取失败
policy = DQN / PPO / GNN-DQN
```

建议顺序：

```text
先实现自适应启发式评分
↓
完成可跑通实验和消融
↓
如果时间够，再加入轻量 RL 版本
```

这样不容易崩。

## 7. 推荐实现路线

### 阶段 1：仿真环境与物体表示

推荐用 PyBullet / Isaac Sim / Mujoco 中较容易上手的一个。

为了稳，建议 PyBullet：

- 安装简单。
- 抓取、碰撞、物体位姿获取方便。
- 可以直接拿 ground-truth pose 做第一版。
- 后续再加入点云估计误差。

输入：

```text
固定物体集合 + 已知模型
```

第一版可以不做复杂识别：

```text
仿真中直接读取 object id 和 pose
```

论文中可以说明真实系统时由点云识别和 pose estimation 提供。

### 阶段 2：抓取候选生成

先不要做复杂 6-DoF 抓取网络。

推荐：

```text
对每个物体基于 CAD / bounding box / 主轴生成若干候选抓取
```

例如：

```text
top grasp
side grasp along x
side grasp along y
diagonal grasp
```

每个 grasp candidate 包含：

```text
grasp center
approach direction
gripper orientation
opening width
```

### 阶段 3：OD 计算

第一版实现：

```text
对目标物体 B 的每个 grasp candidate
沿 approach direction 附近构造若干 rays 或 corridor
检测是否被其他物体 A 阻挡
得到 OD(A -> B)
```

可简化为：

```text
如果 A 的 bounding box / mesh 与 B 的 approach corridor 相交，则 OD 高
距离越远，OD 越低
```

推荐输出：

```text
od_mean(A -> B)
od_max(A -> B)
best_grasp_od(A -> B)
blocked_grasp_ratio(A -> B)
```

### 阶段 4：其他关系特征计算

支撑关系：

```text
A 在 B 下方
A 与 B 接触
A、B 在 xy 平面投影有重叠
→ A supports B
```

高度差：

```text
height_diff(A, B) = z_A - z_B
```

接触关系：

```text
min_distance(A, B) < threshold
```

重叠关系：

```text
xy_overlap_ratio(A, B)
```

可达性影响：

```text
valid_grasps(B with A) / all_grasps(B)
```

或者：

```text
reachability_loss(A -> B)
```

### 阶段 5：自适应顺序决策

每轮：

```text
1. 读取当前场景物体集合
2. 构建关系图
3. 对每个候选目标物体 B 计算 Score(B)
4. 选择得分最高的物体执行抓取
5. 移除物体或仿真执行
6. 更新场景图
7. 重复
```

推荐保留 baseline：

```text
random
nearest-first
top-first / highest-first
OD-only
fixed-weight relation score
adaptive relation score
```

如果做 RL，再加：

```text
RL policy
```

## 8. 实验指标

可以用以下指标：

```text
SR: Success Rate，抓取成功率
CR: Clearance Rate，清空率
ER: Execution Rate，成功执行率
Collision Count，碰撞次数
Average Steps，平均抓取步数
Planning Time，平均规划时间
```

还可以加：

```text
Regrasp Count
Failure Count
Scene Completion Rate
```

消融实验建议：

```text
OD only
OD + support
OD + support + contact
OD + support + contact + reachability
fixed weights
adaptive weights
```

## 9. 论文写法建议

可以这样组织章节：

```text
第 1 章 绪论
第 2 章 相关工作
第 3 章 基于 OD 扩展的物体关系图构建
第 4 章 自适应顺序抓取决策方法
第 5 章 仿真实验与结果分析
第 6 章 总结与展望
```

第 3 章重点：

```text
OD 定义
支撑/接触/高度/可达性关系定义
有向关系图构建
节点特征和边特征
```

第 4 章重点：

```text
抓取顺序问题建模
自适应评分函数
每次抓取后的图更新
如果加入 RL，则介绍 state/action/reward/policy
```

第 5 章重点：

```text
场景设置
baseline
评价指标
消融实验
结果分析
失败案例
```

## 10. 当前最推荐的低风险版本

如果时间有限，建议最终实现：

```text
OD 扩展关系图 + 自适应启发式评分 + 仿真实验
```

不要一开始就承诺完整 GNN+RL。

可以在论文里写：

```text
本文提出一种基于关系图的自适应顺序抓取方法。
强化学习可作为后续扩展，或者作为轻量策略学习模块加入对比实验。
```

如果导师明确要求“智能学习方法”，可以做轻量版：

```text
DQN 输入人工提取的 graph-level features
输出选择哪个物体
```

而不是一上来做复杂 GNN policy。

## 11. 后续给 Linux Codex 的第一句话建议

可以直接对新的 Codex 说：

```text
请先阅读 handoff_od_relation_graph_grasping.md。我的目标是做硕士毕业论文方向：基于 OD 扩展关系图的自适应顺序抓取方法。请帮我把该方案进一步整理成论文开题/技术路线/实现计划，要求容易落地，优先 PyBullet 仿真，不要一开始做复杂 GNN+RL。
```

如果要开始写代码，可以说：

```text
请根据 handoff_od_relation_graph_grasping.md，帮我设计一个 PyBullet 仿真原型项目结构，实现固定物体集合、已知模型、关系图构建、OD 计算和自适应抓取顺序评分的最小可运行版本。
```

## 12. 关键提醒

- OD 是边特征之一，不是所有关系特征的总称。
- edge(A -> B) 应该表示 A 对抓取 B 的影响。
- 视觉遮挡和抓取阻碍是两个概念。
- 支撑关系、接触关系、高度差适合补充 OD 的不足。
- 自适应评分比完整 RL 更稳。
- 后续如果做 RL，最好在启发式方法跑通后再加。
- 论文重点是“关系图建模 + 自适应顺序决策”，不是复现复杂抓取网络。
