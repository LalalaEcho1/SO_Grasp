# 技术路线草案

## 论文主线

关系图建模是结构创新，自适应强化学习是决策创新。

```text
MuJoCo 堆叠场景
-> 物体状态 / 接触 / 支撑 / 高度关系
-> OD 扩展有向关系图
-> 自适应评分 baseline
-> 强化学习策略优化抓取顺序
-> 消融实验和对比实验
```

## 第一个里程碑

完成最小 MuJoCo 原型：

```text
python scripts/run_scene.py --headless
```

能够输出：

- object pose
- contact pair
- directed relation edge
- adaptive baseline ranking

## 第二个里程碑

实现候选抓取与 OD 计算：

```text
object B -> grasp candidates
candidate approach path -> corridor / swept volume
object A intersects corridor -> OD(A -> B)
```

输出：

- best_grasp_od
- blocked_grasp_ratio
- od_mean
- od_max

## 第三个里程碑

把启发式评分变成可复现实验：

- random
- top-first
- OD-only
- fixed relation score
- adaptive relation score

## 第四个里程碑

加入强化学习：

- state: object-level relation features
- action: choose next object
- reward: success, clearance gain, collision penalty, failure penalty
- baseline: DQN or PPO with compact relation features

