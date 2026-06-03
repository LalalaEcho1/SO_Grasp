# YCB 物体选型清单

目标：选择一批在 YCB 中有公开 3D 模型、现实中也容易购买或找到相似替代品的物体，用于后续 MuJoCo 仿真和真实 RGB-D 场景验证。

## 选型原则

1. 现实中容易获得：超市、便利店、文具店或网购容易买到相似形状。
2. 形状先简单后复杂：先用盒子、罐头、瓶子、泡沫块，再加入杯子、工具等非凸物体。
3. 便于堆叠：优先选择稳定、刚性、尺寸适中的物体。
4. 仿真碰撞可控：第一版 collision 用 box / cylinder 近似，visual mesh 后续再接 YCB mesh。
5. 安全：真实实验中避免玻璃、尖锐剪刀、过重清洁剂等高风险物体。

## 第一批：强烈推荐

这些最适合作为下一阶段从简单几何体过渡到真实物体的 starter set。

| YCB 名称 | 现实替代物 | 形状 | 适合原因 | 碰撞近似 |
|---|---|---|---|---|
| `003_cracker_box` | 饼干盒、麦片盒、药盒外包装 | 长方体盒子 | 容易购买，适合堆叠，尺寸稳定 | box |
| `004_sugar_box` | 糖盒、茶叶盒、纸盒包装 | 长方体盒子 | 和 cracker box 类似但尺寸不同，适合测试高度/重叠 | box |
| `008_pudding_box` | 小布丁盒、小食品盒 | 小盒子 | 小尺寸物体，适合测试遮挡和堆叠 | box |
| `009_gelatin_box` | 果冻粉盒、小药盒、小包装盒 | 小盒子 | 轻、小、易买，适合做多物体密集场景 | box |
| `005_tomato_soup_can` | 番茄罐头、饮料罐、八宝粥罐 | 圆柱罐 | 圆柱类典型物体，容易买到替代品 | cylinder |
| `007_tuna_fish_can` | 矮罐头、鱼罐头 | 矮圆柱 | 和高罐形成尺寸差异，适合测试遮挡和接触 | cylinder |
| `010_potted_meat_can` | 午餐肉罐头、方罐 | 矮盒/圆角盒 | 更接近真实非理想盒体，仍然容易近似 | box |
| `061_foam_brick` | 泡沫块、海绵块、EVA 积木 | 规则块 | 安全、轻、适合真实抓取测试 | box |

## 第二批：适合加入但稍复杂

| YCB 名称 | 现实替代物 | 形状 | 主要风险 | 建议 |
|---|---|---|---|---|
| `006_mustard_bottle` | 调味瓶、洗手液瓶、塑料瓶 | 瓶状 | 非均匀截面，重心和碰撞较复杂 | 第二阶段加入，先用 cylinder + box 近似 |
| `011_banana` | 香蕉或仿真香蕉 | 弯曲体 | 非凸、姿态变化大、真实物体易变形 | 不建议第一批 |
| `024_bowl` | 碗 | 凹形物体 | 碰撞体需要凸分解或简化 | 后续用于复杂非凸测试 |
| `025_mug` | 马克杯 | 带把手非凸物体 | 把手碰撞复杂，真实抓取姿态多样 | 后续加入 |
| `035_power_drill` | 电钻/玩具电钻 | 工具 | 形状复杂，真实获取和安全性一般 | 后期复杂测试 |
| `036_wood_block` | 木块 | 规则块 | 容易获取，但和 foam brick 功能重复 | 可选 |

## 暂不建议

| YCB 名称 | 原因 |
|---|---|
| `021_bleach_cleanser` | 真实物体较重且液体容器不安全，仿真可用但真实测试不优先 |
| `037_scissors` | 真实测试不安全，形状细长且碰撞复杂 |
| `040_large_marker` | 细长圆柱容易滚动，后续可作为挑战物体 |
| `051_large_clamp` / `052_extra_large_clamp` | 形状复杂，不适合 starter set |

## 建议执行顺序

### 阶段 1：YCB-Lite 规则碰撞版本

先不用导入复杂 mesh，直接根据 YCB 类别建立简化 collision：

```text
box 类：cracker_box / sugar_box / pudding_box / gelatin_box / foam_brick
cylinder 类：tomato_soup_can / tuna_fish_can
rounded-box 类：potted_meat_can
```

目标：

```text
替换当前简单 box/cylinder 场景
保留真实物体类别名
先跑通 OD、关系图、顺序决策
```

项目中对应场景文件：

```text
assets/scenes/ycb_lite_stacked.xml
```

运行：

```bash
python scripts/run_scene.py --scene assets/scenes/ycb_lite_stacked.xml --headless
python scripts/export_visuals.py --scene assets/scenes/ycb_lite_stacked.xml --gl-backend osmesa
```

### 阶段 2：导入 YCB visual mesh

使用 YCB mesh 作为 visual 外观，collision 仍用简化几何体。

目标：

```text
截图和论文图更真实
仿真仍然稳定
```

项目中对应脚本：

```text
scripts/download_ycb_meshes.py
scripts/generate_ycb_mesh_scene.py
assets/scenes/ycb_mesh_stacked.xml
```

执行：

```bash
python scripts/download_ycb_meshes.py --objects starter
python scripts/run_scene.py --scene assets/scenes/ycb_mesh_stacked.xml --headless
python scripts/export_visuals.py --scene assets/scenes/ycb_mesh_stacked.xml --gl-backend osmesa
```

说明：

```text
download_ycb_meshes.py 会下载官方 YCB Google 16k textured.obj / texture_map.png
generate_ycb_mesh_scene.py 会读取 OBJ 顶点范围，自动计算 visual mesh 的单位缩放、居中偏移和 90 度朝向匹配
MuJoCo 中 collision 使用 mesh-aligned box / cylinder 近似，用于稳定接触、支撑和 OD 计算
official mesh 用于 visual，使截图和后续论文图更接近真实物体
```

### 阶段 3：复杂碰撞体

对 mug、bowl、banana、mustard_bottle 等物体尝试：

```text
convex decomposition
multi-geom approximation
mesh collision
```

目标：

```text
提升复杂物体堆叠与真实场景一致性
```

## 后续真实场景购买建议

优先买或找相似物，不需要完全同品牌：

```text
长方体饼干盒 2 个
小食品盒 2 个
圆柱罐头 2 个
矮罐头 1 个
泡沫块 / 海绵块 2-3 个
塑料调味瓶 1 个
马克杯 1 个，第二阶段用
```

真实测试时先贴不同颜色标签或使用颜色差异明显的物体，降低感知分割难度。
