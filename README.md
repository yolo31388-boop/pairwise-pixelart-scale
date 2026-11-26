# pairwise-pixelart-scale

## 领域

游戏、图形与多媒体

## 主题

像素画整数缩放

## 说明

整数倍、跨块邻域与全局色板。

`imaging/pixelart_scale.py` 已实现整数最近邻/自适应 EPX 缩放、跨块邻域、
全局色板量化、尺寸补齐、导出和边缘校验。公开入口对非法输入返回
`{"ok": False, ...}` 报告而不是抛出异常。

运行：

```bash
python -m pytest -q
```
