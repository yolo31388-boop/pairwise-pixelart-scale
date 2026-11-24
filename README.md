# pairwise-pixelart-scale

## 领域

游戏、图形与多媒体

## 主题

像素画整数缩放

## 说明

整数倍、跨块邻域与全局色板。

初始状态下 `imaging/pixelart_scale.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
