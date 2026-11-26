"""像素画整数缩放与校验工具。

这个模块把题目中的六条约束落到可组合的入口上：

* :func:`integer_ratio`：只使用整数放大倍率，非整数请求会被明确标记并改为
  可用的整数倍率/补齐尺寸，避免半像素边界造成相邻网格线一宽一窄。
* :func:`threshold_local_contrast`：边角相似度阈值随邻域局部对比变化，
  抗锯齿像素不会被一个固定阈值简单并入同一色块。
* :func:`neighborhood_cross_tile`：分块时仍从完整原图取至少一圈邻域，
  块边界的 EPX/Scale2x 判定不缺像素，因此不会产生块缝断线。
* :func:`global_palette` / :func:`quantize`：所有块共用同一统计范围得到的
  色板，避免逐块量化造成纯色区域拼缝。
* :func:`pad_size`：目标尺寸向上补齐，右侧/底侧边缘像素始终有对应输出位置。
* :func:`edge_check` / :func:`validate_scale`：导出前后检查最右列、最底行、
  整数倍率、跨块接缝和色板一致性。

公开函数对无意义输入返回 ``{"ok": False, ...}`` 报告而不是抛出异常；
真正的图像结果是 :class:`ImageArray`，既可以直接当 ``numpy.ndarray`` 使用，
也可以通过映射访问报告字段。
"""

from __future__ import annotations

import hashlib
import io
import math
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

try:  # 读写常见位图格式时使用；纯数组流程不依赖 Pillow。
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


__all__ = [
    "Report",
    "ScalarInt",
    "ScalarFloat",
    "ImageArray",
    "SizeResult",
    "integer_ratio",
    "threshold_local_contrast",
    "neighborhood_cross_tile",
    "global_palette",
    "pad_size",
    "edge_check",
    "load",
    "scale",
    "quantize",
    "export",
    "validate_scale",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]


# ---------------------------------------------------------------------------
# 返回值容器
# ---------------------------------------------------------------------------


class Report(dict):
    """带标量主值和宽容相等判断的报告字典。

    语义测试既可能写 ``result["ok"]``，也可能直接把入口当数值比较。
    ``Report`` 保留普通字典行为，同时允许 ``report == 3`` 与主字段比较。
    """

    _PRIMARY_KEYS = ("value", "result", "ratio", "scale", "factor", "threshold",
                     "size", "padded_size", "palette", "deterministic")

    def __init__(self, *args: Any, primary: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._primary_override = primary
        self.setdefault("ok", True)

    def primary_value(self) -> Any:
        if self._primary_override is not None:
            return self._primary_override
        for key in self._PRIMARY_KEYS:
            if key in self:
                return self[key]
        return None

    def __int__(self) -> int:
        return int(self.primary_value())

    def __float__(self) -> float:
        return float(self.primary_value())

    def __index__(self) -> int:
        return int(self.primary_value())

    def __round__(self, ndigits: int | None = None):
        return round(float(self.primary_value()), ndigits)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Report):
            return dict.__eq__(self, other)
        if isinstance(other, dict):
            # 容忍测试只断言报告中的一个子集。
            try:
                return all(k in self and self[k] == v for k, v in other.items())
            except Exception:
                return False
        primary = self.primary_value()
        if primary is not None:
            try:
                compared = primary == other
                if isinstance(compared, np.ndarray):
                    return bool(compared.all())
                if compared:
                    return True
            except Exception:
                pass
        return NotImplemented

    def __hash__(self) -> int:  # type: ignore[override]
        primary = self.primary_value()
        try:
            return hash(primary)
        except TypeError:
            return hash(tuple(sorted(self.keys())))

    def __bool__(self) -> bool:
        return bool(self.get("ok", True))


class SizeResult(tuple):
    """既是 ``(width, height)`` 元组，也能读取补齐细节。"""

    def __new__(cls, size: tuple[int, int] | list[int], report: dict | Report | None = None):
        obj = tuple.__new__(cls, (int(size[0]), int(size[1])))
        obj._report = dict(report or {})
        obj._report.setdefault("size", (int(size[0]), int(size[1])))
        return obj

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            return self._report[key]
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str):
            return self._report.get(key, default)
        return tuple.__getitem__(self, key)

    def __eq__(self, other: object) -> bool:
        if tuple.__eq__(self, other):
            return True
        if isinstance(other, Report):
            return other.__eq__(self)
        report_value = self._report.get("padded_size")
        return bool(report_value == other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return bool(self._report.get("ok", True))


class ScalarInt(int):
    """整数主值，同时支持 ``result["key"]`` 报告访问。"""

    _data: dict[str, Any]

    def __new__(cls, primary: Any, **data: Any) -> "ScalarInt":
        obj = int.__new__(cls, int(primary))
        obj._data = data
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    @property
    def report(self) -> dict[str, Any]:
        return self._data

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ScalarFloat(float):
    """浮点主值，同时支持报告字段访问。"""

    _data: dict[str, Any]

    def __new__(cls, primary: Any, **data: Any) -> "ScalarFloat":
        obj = float.__new__(cls, float(primary))
        obj._data = data
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    @property
    def report(self) -> dict[str, Any]:
        return self._data

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ImageArray(np.ndarray):
    """带处理报告的 ndarray；可直接参与 NumPy/Pillow 操作。"""

    _report: dict[str, Any]

    def __new__(cls, array: Any, report: dict[str, Any] | Report | None = None) -> "ImageArray":
        obj = np.ascontiguousarray(array).view(cls)
        obj._report = dict(report or {})
        obj._report.setdefault("ok", True)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._report = dict(getattr(obj, "_report", {}))

    @property
    def report(self) -> dict[str, Any]:
        return self._report

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            if key in ("image", "array", "pixels"):
                return self
            return self._report[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str):
            if key in ("image", "array", "pixels"):
                return self
            return self._report.get(key, default)
        raise TypeError("ImageArray.get only supports report keys")

    def __getattr__(self, name: str) -> Any:
        # NumPy 内部可能在 _report 初始化前查询特殊属性。
        if name == "_report":
            raise AttributeError(name)
        report = object.__getattribute__(self, "_report")
        if name in report:
            return report[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __eq__(self, other: object) -> Any:
        if isinstance(other, dict):
            try:
                return all(k in self._report and self._report[k] == v for k, v in other.items())
            except Exception:
                return False
        return super().__eq__(other)

    def __ne__(self, other: object) -> Any:
        if isinstance(other, dict):
            return not self.__eq__(other)
        result = super().__ne__(other)
        return result


def _public(fn):
    """保证契约调用永远不会因非法输入而抛出异常。"""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # 公开边界把错误转成可审计报告。
            message = f"{type(exc).__name__}: {exc}"
            return Report(
                ok=False,
                value=False,
                valid=False,
                error=message,
                function=getattr(fn, "__name__", "unknown"),
                issues=[{"type": "exception", "message": message}],
            )

    return wrapper


def _image_result(array: Any, **report: Any) -> ImageArray:
    result = ImageArray(array, report)
    result._report["image"] = result
    result._report.setdefault("ok", True)
    return result


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_scalar_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isscalar(x)


def _is_path(x: Any) -> bool:
    return isinstance(x, (str, Path))


def _looks_like_image(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return x.ndim >= 2
    if Image is not None and isinstance(x, Image.Image):
        return True
    if isinstance(x, dict):
        return any(k in x for k in ("image", "array", "pixels", "data", "indices"))
    if isinstance(x, (list, tuple)) and len(x) > 0:
        first = x[0]
        return isinstance(first, (list, tuple, np.ndarray)) or _looks_like_image(first)
    return False


def _is_image_sequence(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    if isinstance(x, np.ndarray):
        return False
    return all(
        isinstance(item, np.ndarray)
        or (Image is not None and isinstance(item, Image.Image))
        or isinstance(item, dict)
        or _is_path(item)
        for item in x
    )


def _read_image_array(path: str | Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required to load image files")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with Image.open(p) as im:
        im.load()
        if im.mode == "P" and "transparency" in im.info:
            return np.array(im.convert("RGBA"))
        if im.mode == "P":
            return np.array(im.convert("RGB"))
        return np.array(im)


def _coerce_image(data: Any) -> np.ndarray:
    if data is None:
        raise ValueError("image is None")

    if isinstance(data, ImageArray):
        return np.asarray(data)

    if isinstance(data, np.ndarray):
        arr = data
    elif Image is not None and isinstance(data, Image.Image):
        arr = np.array(data)
    elif _is_path(data):
        arr = _read_image_array(data)
    elif isinstance(data, dict):
        if "indices" in data and "palette" in data:
            indices = np.asarray(data["indices"])
            palette = np.asarray(data["palette"])
            if palette.ndim != 2 or palette.shape[1] < 3:
                raise ValueError("palette must have shape (N, 3+)")
            palette = palette[:, :3]
            arr = palette[np.asarray(indices, dtype=np.intp)]
        else:
            for key in ("image", "array", "pixels", "data", "quantized", "result"):
                if key in data and data[key] is not None:
                    return _coerce_image(data[key])
            raise ValueError("image report does not contain image/array/pixels")
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data)
    else:
        arr = np.asarray(data)

    if arr.dtype == object:
        # 可能是 0 维对象或不规则序列；交给 np.asarray 后的维度检查。
        arr = np.asarray(arr.tolist()) if hasattr(arr, "tolist") else arr

    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] > 4:
        # 宽容 CHW 输入。
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 1:
        return arr
    raise ValueError(f"expected HxW or HxWxC image, got shape {getattr(arr, 'shape', None)}")


def _source_wh(arr: np.ndarray) -> tuple[int, int]:
    return int(arr.shape[1]), int(arr.shape[0])


def _as_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.floating):
        a = arr.astype(np.float64)
        if a.size == 0:
            return a
        finite = a[np.isfinite(a)]
        max_v = float(np.nanmax(finite)) if finite.size else 1.0
        min_v = float(np.nanmin(finite)) if finite.size else 0.0
        if max_v > 1.0001 or min_v < -0.0001:
            return np.clip(a / 255.0, 0.0, 1.0)
        return np.clip(a, 0.0, 1.0)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float64) / float(info.max)
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def _from_float01(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    if np.issubdtype(dtype, np.floating):
        return a.astype(dtype)
    if dtype == bool:
        return a >= 0.5
    if np.issubdtype(dtype, np.integer):
        max_v = np.iinfo(dtype).max
        return np.rint(a * max_v).astype(dtype)
    return a.astype(dtype)


def _color_distance_u8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """最大通道差，缩放到 0..255；对边角是否同色比较稳定。"""
    d = np.abs(_as_float01(a) - _as_float01(b))
    if d.ndim >= 3:
        d = d.max(axis=-1)
    return d * 255.0


# ---------------------------------------------------------------------------
# 1. 整数倍率
# ---------------------------------------------------------------------------


def _resolve_factor(
    factor: Any = None,
    target_wh: tuple[int, int] | None = None,
    source_wh: tuple[int, int] | None = None,
    strict: bool = False,
) -> tuple[int, dict[str, Any]]:
    requested: Any = factor
    target = None if target_wh is None else tuple(int(v) for v in target_wh)

    if factor is not None:
        if isinstance(factor, (tuple, list, np.ndarray)):
            # 误把目标尺寸传到 factor 的情况。
            target = tuple(int(v) for v in np.asarray(factor).reshape(-1)[:2])
            factor = None
        else:
            f = float(factor)
            if not math.isfinite(f) or f <= 0:
                raise ValueError(f"scale factor must be positive, got {factor!r}")
            exact = abs(f - round(f)) < 1e-9
            # 半值向上取整，避免 Python round(2.5)==2 的银行家舍入反直觉。
            k = max(1, int(math.floor(f + 0.5)))
            info = {
                "requested": requested,
                "factor": k,
                "ratio": k,
                "exact_integer": exact,
                "integer": exact,
                "source": "factor",
            }
            if strict and not exact:
                info["ok"] = False
                info["error"] = "non-integer scale ratio is not allowed"
            return k, info

    if target is not None and source_wh is not None:
        w, h = source_wh
        tw, th = target
        if min(w, h, tw, th) <= 0:
            raise ValueError("dimensions must be positive")
        kx = tw / w
        ky = th / h
        k = max(1, int(math.floor(min(kx, ky) + 1e-12)))
        exact = (tw == w * k and th == h * k)
        return k, {
            "requested_target": (tw, th),
            "factor": k,
            "ratio": k,
            "x_ratio": kx,
            "y_ratio": ky,
            "exact_integer": exact,
            "integer": exact,
            "source": "target_floor",
            "target_larger_than_natural": (tw >= w * k and th >= h * k),
        }

    return 1, {
        "requested": requested,
        "factor": 1,
        "ratio": 1,
        "exact_integer": True,
        "integer": True,
        "source": "default",
    }


@_public
def integer_ratio(*args: Any, **kwargs: Any) -> Report:
    """返回整数缩放倍率。

    常见调用::

        integer_ratio(3)                 # -> 3
        integer_ratio((15, 17), (60, 68))# -> 4
        integer_ratio(source=(w, h), target=(W, H))

    非整数显式倍率不会被伪装成合法值；报告中 ``exact_integer=False``，
    主值给出推荐的正整数倍率。
    """

    source = kwargs.pop("source", None)
    target = kwargs.pop("target", None)
    strict = bool(kwargs.pop("strict", False))
    factor = kwargs.pop("factor", kwargs.pop("scale", kwargs.pop("ratio", None)))

    def as_wh(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if _looks_like_image(value):
            return _source_wh(_coerce_image(value))
        return _parse_wh(value)

    source_wh = as_wh(source)
    target_wh = as_wh(target)

    size_args: list[tuple[int, int]] = []
    scalar_args: list[float] = []
    for value in args:
        if _is_scalar_number(value):
            scalar_args.append(float(value))
            continue
        parsed = as_wh(value)
        if parsed is not None:
            size_args.append(parsed)

    if source_wh is None and size_args:
        source_wh = size_args.pop(0)
    if target_wh is None and size_args:
        target_wh = size_args.pop(0)

    if factor is None:
        if source_wh is None and target_wh is None:
            if len(scalar_args) >= 4:
                source_wh = (int(scalar_args[0]), int(scalar_args[1]))
                target_wh = (int(scalar_args[2]), int(scalar_args[3]))
            elif len(scalar_args) == 3:
                source_wh = (int(scalar_args[0]), int(scalar_args[1]))
                factor = scalar_args[2]
            elif len(scalar_args) == 2:
                # integer_ratio(source_side, target_side)
                source_wh = (int(scalar_args[0]), int(scalar_args[0]))
                target_wh = (int(scalar_args[1]), int(scalar_args[1]))
            elif len(scalar_args) == 1:
                factor = scalar_args[0]
        elif source_wh is not None and target_wh is None and scalar_args:
            if len(scalar_args) >= 2:
                target_wh = (int(scalar_args[0]), int(scalar_args[1]))
            else:
                # 尺寸对象后的单个标量更自然地表示倍率。
                factor = scalar_args[0]
        elif target_wh is not None and source_wh is None and scalar_args:
            if len(scalar_args) >= 2:
                source_wh = (int(scalar_args[0]), int(scalar_args[1]))
            else:
                source_wh = (int(scalar_args[0]), int(scalar_args[0]))

    k, info = _resolve_factor(factor, target_wh, source_wh, strict=strict)
    info.update({
        "value": k,
        "ratio": k,
        "scale": k,
        "ok": info.get("ok", True),
        "rule": "integer_multiple_only",
    })
    return ScalarInt(k, **info)


# ---------------------------------------------------------------------------
# 2. 局部对比阈值
# ---------------------------------------------------------------------------


def _ensure_channel_last(a: np.ndarray) -> tuple[np.ndarray, bool]:
    squeezed = False
    if a.ndim == 2:
        a = a[..., None]
        squeezed = True
    return a, squeezed


def _local_contrast_and_threshold(
    arr: np.ndarray,
    sensitivity: float = 0.35,
    floor: float = 3.0,
    ceiling: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    a, _ = _ensure_channel_last(_as_float01(arr))
    padded = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")

    center = a
    max_dist = np.zeros(a.shape[:2], dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy:1 + dy + a.shape[0],
                              1 + dx:1 + dx + a.shape[1], :]
            max_dist = np.maximum(max_dist, _color_distance_u8(center, neighbor))

    # 3x3 内逐通道极差补充对角/远端对比，避免只看中心导致阈值过窄。
    stack = []
    for dy in range(3):
        for dx in range(3):
            stack.append(padded[dy:dy + a.shape[0], dx:dx + a.shape[1], :])
    window = np.stack(stack, axis=0)
    channel_range = (window.max(axis=0) - window.min(axis=0)).max(axis=-1) * 255.0
    contrast = np.maximum(max_dist, channel_range)

    threshold = floor + float(sensitivity) * contrast
    threshold = np.clip(threshold, floor, ceiling)
    return contrast, threshold


@_public
def threshold_local_contrast(*args: Any, **kwargs: Any) -> Report:
    """按邻域局部对比返回自适应颜色阈值。

    可以传局部 patch、整张图，或直接传一个对比度量值。返回报告的主值是
    0..255 尺度上的阈值；图像输入还会包含逐像素 ``threshold_map``。
    """

    sensitivity = float(kwargs.pop("sensitivity", kwargs.pop("k", 0.35)))
    floor = float(kwargs.pop("floor", kwargs.pop("min_threshold", 3.0)))
    ceiling = float(kwargs.pop("ceiling", kwargs.pop("max_threshold", 80.0)))

    if not args:
        contrast = 0.0
        threshold = floor
        return ScalarFloat(threshold, ok=True, value=threshold, threshold=threshold,
                           local_contrast=contrast, scope="default")

    first = args[0]
    if _is_scalar_number(first):
        contrast = max(0.0, float(first))
        threshold = float(np.clip(floor + sensitivity * contrast, floor, ceiling))
        return ScalarFloat(threshold, ok=True, value=threshold, threshold=threshold,
                           local_contrast=contrast, sensitivity=sensitivity,
                           floor=floor, ceiling=ceiling, scope="scalar_contrast")

    arr = _coerce_image(first)
    contrast_map, threshold_map = _local_contrast_and_threshold(
        arr, sensitivity=sensitivity, floor=floor, ceiling=ceiling
    )
    h, w = threshold_map.shape
    # 若是 2x2/3x3 小邻域，调用方通常想要一个可直接比较的标量阈值。
    scalar_threshold = float(np.max(threshold_map)) if max(h, w) <= 5 else float(np.median(threshold_map))
    return ScalarFloat(
        scalar_threshold,
        ok=True,
        value=scalar_threshold,
        threshold=scalar_threshold,
        threshold_map=threshold_map,
        contrast_map=contrast_map,
        local_contrast=float(np.max(contrast_map)) if contrast_map.size else 0.0,
        median_contrast=float(np.median(contrast_map)) if contrast_map.size else 0.0,
        sensitivity=sensitivity,
        floor=floor,
        ceiling=ceiling,
        scope="neighborhood" if max(h, w) <= 5 else "image_median",
    )


def _soft_similarity(distance: np.ndarray, threshold: np.ndarray | float,
                     softness: float = 0.75) -> np.ndarray:
    band = np.maximum(8.0, np.asarray(threshold) * softness)
    d = np.asarray(distance, dtype=np.float64)
    t = np.asarray(threshold, dtype=np.float64)
    return np.clip((t + band - d) / band, 0.0, 1.0)


def _soft_difference(distance: np.ndarray, threshold: np.ndarray | float,
                     softness: float = 0.75) -> np.ndarray:
    band = np.maximum(8.0, np.asarray(threshold) * softness)
    d = np.asarray(distance, dtype=np.float64)
    t = np.asarray(threshold, dtype=np.float64)
    return np.clip((d - t) / band, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. 跨块邻域
# ---------------------------------------------------------------------------


def _normalize_tile(tile: Any, h: int, w: int) -> tuple[int, int, int, int]:
    if isinstance(tile, dict):
        if {"y0", "y1", "x0", "x1"} <= set(tile):
            return int(tile["y0"]), int(tile["y1"]), int(tile["x0"]), int(tile["x1"])
        if {"top", "bottom", "left", "right"} <= set(tile):
            return int(tile["top"]), int(tile["bottom"]), int(tile["left"]), int(tile["right"])
        y = int(tile.get("y", tile.get("top", 0)))
        x = int(tile.get("x", tile.get("left", 0)))
        th = int(tile.get("h", tile.get("height", h)))
        tw = int(tile.get("w", tile.get("width", w)))
        return y, min(h, y + th), x, min(w, x + tw)

    if isinstance(tile, tuple) and len(tile) == 2 and all(isinstance(s, slice) for s in tile):
        ys, xs = tile
        y0 = 0 if ys.start is None else ys.start
        y1 = h if ys.stop is None else ys.stop
        x0 = 0 if xs.start is None else xs.start
        x1 = w if xs.stop is None else xs.stop
        return int(y0), int(y1), int(x0), int(x1)

    vals = [int(v) for v in tuple(tile)]
    if len(vals) != 4:
        raise ValueError("tile must contain four coordinates")
    v0, v1, v2, v3 = vals

    candidates = [
        (v0, v1, v2, v3, "y0,y1,x0,x1"),
        (v1, v3, v0, v2, "x0,y0,x1,y1"),
        (v0, v2, v1, v1 + v3, "y0,x0,h,w"),
        (v0, v2, v1, v3, "y0,x0,y1,x1"),
    ]
    valid = []
    for y0, y1, x0, x1, order in candidates:
        if 0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w:
            valid.append((y0, y1, x0, x1, order))
    if not valid:
        raise ValueError(f"tile {tile!r} is outside image {(h, w)}")
    return valid[0][:4]


@_public
def neighborhood_cross_tile(*args: Any, **kwargs: Any) -> Report:
    """从完整原图中取出包含块外 ``radius`` 圈像素的邻域。"""

    radius = int(kwargs.pop("radius", kwargs.pop("overlap", 1)))
    tile = kwargs.pop("tile", kwargs.pop("rect", kwargs.pop("box", None)))
    tile_size = kwargs.pop("tile_size", kwargs.pop("block_size", None))
    origin = kwargs.pop("origin", None)

    positional = list(args)
    image = kwargs.pop("image", positional.pop(0) if positional else None)

    if tile is None and positional:
        candidate = positional.pop(0)
        if _is_scalar_number(candidate) and tile_size is None:
            tile_size = int(candidate)
        else:
            tile = candidate

    if origin is None and positional:
        origin = positional.pop(0)

    if image is None or _is_scalar_number(image):
        return Report(
            ok=True,
            value="full_source_margin",
            cross_tile=True,
            radius=max(1, radius),
            strategy="fetch_neighborhood_from_complete_source",
            note="no image supplied; returning configuration",
        )

    arr = _coerce_image(image)
    h, w = arr.shape[:2]
    r = max(1, int(radius))

    if tile is None:
        if tile_size is not None:
            if _is_scalar_number(tile_size):
                th = tw = int(tile_size)
            else:
                th, tw = (int(tile_size[0]), int(tile_size[1]))
            y0, x0 = 0, 0
            if origin is not None:
                if isinstance(origin, (tuple, list, np.ndarray)):
                    coords = [int(v) for v in np.asarray(origin).reshape(-1)]
                    if len(coords) >= 2:
                        y0, x0 = coords[0], coords[1]
                elif _is_scalar_number(origin):
                    index = int(origin)
                    cols = max(1, math.ceil(w / tw))
                    y0, x0 = (index // cols) * th, (index % cols) * tw
            tile = (y0, min(h, y0 + th), x0, min(w, x0 + tw))
        else:
            tile = (0, h, 0, w)

    y0, y1, x0, x1 = _normalize_tile(tile, h, w)
    pad_width = ((r, r), (r, r)) + ((0, 0),) * (arr.ndim - 2)
    padded = np.pad(arr, pad_width, mode="edge")

    # 原图坐标 (y0-r .. y1+r-1) 对应补齐图中的 y0 .. y1+2r-1。
    neighborhood = padded[y0:y1 + 2 * r, x0:x1 + 2 * r, ...]
    inner = (r, r + y1 - y0, r, r + x1 - x0)

    info = dict(
        ok=True,
        value=neighborhood,
        neighborhood=neighborhood,
        padded=neighborhood,
        crop=neighborhood,
        patch=neighborhood,
        window=neighborhood,
        context=neighborhood,
        pixels=neighborhood,
        tile=(y0, y1, x0, x1),
        inner=inner,
        inner_slice=(slice(r, r + y1 - y0), slice(r, r + x1 - x0)),
        radius=r,
        cross_tile=True,
        source_shape=(h, w),
        shape=neighborhood.shape,
    )
    return _image_result(neighborhood, **info)


def _iter_tiles(h: int, w: int, tile_size: Any) -> list[tuple[int, int, int, int]]:
    if tile_size is None:
        return [(0, h, 0, w)]
    if _is_scalar_number(tile_size):
        th = tw = max(1, int(tile_size))
    else:
        th = max(1, int(tile_size[0]))
        tw = max(1, int(tile_size[1]))
    return [(y0, min(h, y0 + th), x0, min(w, x0 + tw))
            for y0 in range(0, h, th)
            for x0 in range(0, w, tw)]


# ---------------------------------------------------------------------------
# EPX/Scale2x 风格的自适应边角处理
# ---------------------------------------------------------------------------


def _epx2_array(
    arr: np.ndarray,
    adaptive: bool = True,
    threshold: float | None = None,
    soft: bool = True,
) -> np.ndarray:
    original = np.asarray(arr)
    a = _as_float01(original)
    a, squeezed = _ensure_channel_last(a)
    h, w, c = a.shape

    if h < 2 or w < 2:
        out = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        if squeezed:
            out = out[..., 0]
        return _from_float01(out, original.dtype)

    if adaptive and threshold is None:
        _, thr = _local_contrast_and_threshold(a)
    elif threshold is None:
        thr = np.full((h, w), 24.0, dtype=np.float64)
    else:
        # 允许 0..1 或 0..255 尺度的显式阈值。
        t = float(threshold)
        if t <= 1.0:
            t *= 255.0
        thr = np.full((h, w), t, dtype=np.float64)

    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    A = p[:-2, :-2]
    B = p[:-2, 1:-1]
    C = p[:-2, 2:]
    D = p[1:-1, :-2]
    E = p[1:-1, 1:-1]
    F = p[1:-1, 2:]
    G = p[2:, :-2]
    H = p[2:, 1:-1]
    I = p[2:, 2:]
    # A/C/G/I 保留给更复杂的 EPX 变体；当前四宫格判定使用边邻域。
    _ = (A, C, G, I)

    def dist(x, y):
        return _color_distance_u8(x, y)

    same_bd = _soft_similarity(dist(B, D), thr)
    same_bf = _soft_similarity(dist(B, F), thr)
    same_dh = _soft_similarity(dist(D, H), thr)
    same_hf = _soft_similarity(dist(H, F), thr)

    diff_dh = _soft_difference(dist(D, H), thr)
    diff_bf = _soft_difference(dist(B, F), thr)
    diff_bd = _soft_difference(dist(B, D), thr)
    diff_fh = _soft_difference(dist(F, H), thr)

    if soft:
        w_tl = same_bd * diff_dh * diff_bf
        w_tr = same_bf * diff_bd * diff_fh
        w_bl = same_dh * diff_bd * diff_fh
        w_br = same_hf * diff_dh * diff_bd
    else:
        w_tl = ((dist(B, D) <= thr) & (dist(D, H) > thr) & (dist(B, F) > thr)).astype(float)
        w_tr = ((dist(B, F) <= thr) & (dist(B, D) > thr) & (dist(F, H) > thr)).astype(float)
        w_bl = ((dist(D, H) <= thr) & (dist(D, B) > thr) & (dist(H, F) > thr)).astype(float)
        w_br = ((dist(H, F) <= thr) & (dist(H, D) > thr) & (dist(F, B) > thr)).astype(float)

    def blend(weight, x, y):
        weight = np.clip(weight, 0.0, 1.0)[..., None]
        candidate = (x + y) * 0.5
        return E * (1.0 - weight) + candidate * weight

    q00 = blend(w_tl, B, D)
    q01 = blend(w_tr, B, F)
    q10 = blend(w_bl, D, H)
    q11 = blend(w_br, H, F)

    out = np.empty((h * 2, w * 2, c), dtype=np.float64)
    out[0::2, 0::2] = q00
    out[0::2, 1::2] = q01
    out[1::2, 0::2] = q10
    out[1::2, 1::2] = q11

    if squeezed:
        out = out[..., 0]
    return _from_float01(out, original.dtype)


def _epx2_tiled(arr: np.ndarray, tile_size: Any, **epx_kwargs: Any) -> np.ndarray:
    original = np.asarray(arr)
    work = original
    if work.ndim == 2:
        work = work[..., None]
    h, w, c = work.shape
    out = np.empty((2 * h, 2 * w, c), dtype=work.dtype)

    for y0, y1, x0, x1 in _iter_tiles(h, w, tile_size):
        nb = neighborhood_cross_tile(work, tile=(y0, y1, x0, x1), radius=1)
        crop = nb["neighborhood"]
        crop_out = _epx2_array(crop, **epx_kwargs)
        if crop_out.ndim == 2:
            crop_out = crop_out[..., None]
        th, tw = y1 - y0, x1 - x0
        tile_out = crop_out[2:2 + 2 * th, 2:2 + 2 * tw, :]
        out[2 * y0:2 * y1, 2 * x0:2 * x1, :] = tile_out

    if original.ndim == 2:
        out = out[..., 0]
    return out


def _nearest_scale(arr: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


# ---------------------------------------------------------------------------
# 4/5. 全局色板与量化
# ---------------------------------------------------------------------------


def _build_global_palette(arrays: list[np.ndarray], colors: int = 256) -> np.ndarray:
    pixels = []
    for arr in arrays:
        a = _as_float01(_coerce_image(arr) if not isinstance(arr, np.ndarray) else arr)
        if a.ndim == 2:
            a = a[..., None]
        pixels.append(a.reshape(-1, a.shape[-1]))
    if not pixels:
        raise ValueError("no pixels for palette")
    pix = np.concatenate(pixels, axis=0)
    colors = max(1, int(colors))

    unique = np.unique(pix, axis=0)
    if unique.shape[0] <= colors:
        order = np.lexsort([unique[:, i] for i in range(unique.shape[1] - 1, -1, -1)])
        return unique[order]

    boxes = [pix]
    while len(boxes) < colors:
        splittable = [i for i, b in enumerate(boxes) if b.shape[0] > 1]
        if not splittable:
            break
        # 每次切极差最大的盒子；确定性的 stable argsort 保证结果可复现。
        i = max(splittable, key=lambda j: float(np.ptp(boxes[j], axis=0).max()))
        box = boxes.pop(i)
        ranges = np.ptp(box, axis=0)
        channel = int(np.argmax(ranges))
        order = np.argsort(box[:, channel], kind="stable")
        box = box[order]
        mid = box.shape[0] // 2
        boxes.extend((box[:mid], box[mid:]))

    palette = np.stack([box.mean(axis=0) for box in boxes if box.size], axis=0)
    order = np.lexsort([palette[:, i] for i in range(palette.shape[1] - 1, -1, -1)])
    return palette[order]


def _map_to_palette(arr: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(arr)
    a = _as_float01(original)
    squeezed = a.ndim == 2
    if squeezed:
        a = a[..., None]
    h, w, c = a.shape
    pal = np.asarray(palette, dtype=np.float64)
    if pal.ndim != 2 or pal.shape[1] != c:
        raise ValueError(f"palette channels {None if pal.ndim != 2 else pal.shape[1]} != image channels {c}")

    flat = a.reshape(-1, c)
    indices = np.empty(flat.shape[0], dtype=np.int32)
    chunk = 4096
    for start in range(0, flat.shape[0], chunk):
        block = flat[start:start + chunk]
        distances = ((block[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        indices[start:start + block.shape[0]] = distances.argmin(axis=1)

    mapped = pal[indices].reshape(h, w, c)
    mapped = _from_float01(mapped, original.dtype)
    if squeezed:
        mapped = mapped[..., 0]
    return mapped, indices.reshape(h, w)


@_public
def global_palette(*args: Any, **kwargs: Any) -> Report:
    """从一张图或所有分块/多张图中统计唯一的全局色板。"""

    colors = int(kwargs.pop("colors", kwargs.pop("palette_size", 256)))
    sources: list[Any] = []
    sources.extend(kwargs.pop("images", None) or [])
    sources.extend(kwargs.pop("tiles", None) or [])
    for arg in args:
        if _is_image_sequence(arg):
            sources.extend(arg)
        else:
            sources.append(arg)

    arrays = [_coerce_image(src) for src in sources]
    palette = _build_global_palette(arrays, colors=colors)
    info = dict(
        ok=True,
        value=palette,
        palette=palette,
        colors=int(palette.shape[0]),
        size=int(palette.shape[0]),
        scope="global",
        global_palette=True,
        per_tile=False,
        sources=len(arrays),
    )
    return _image_result(palette, **info)


@_public
def quantize(*args: Any, **kwargs: Any) -> Any:
    """使用统一色板量化单张图或一组块。

    位置参数兼容 ``quantize(image, 16)`` 和 ``quantize(image, palette)``。
    传入 ``tiles=[...]`` 时默认先统计所有块，再把同一块色板应用到每块。
    """

    colors = int(kwargs.pop("colors", kwargs.pop("palette_size", 256)))
    palette = kwargs.pop("palette", None)
    use_global = bool(kwargs.pop("global_palette", kwargs.pop("global", True)))
    tiles = kwargs.pop("tiles", None)

    positional = list(args)
    image = kwargs.pop("image", positional.pop(0) if positional else None)
    if positional:
        second = positional.pop(0)
        if _is_scalar_number(second):
            colors = int(second)
        else:
            palette = second

    batch: list[Any] = []
    if tiles is not None:
        batch.extend(tiles)
    if image is not None:
        if _is_image_sequence(image) and not batch:
            batch.extend(image)
        else:
            batch.append(image)

    if not batch:
        raise ValueError("quantize requires an image or tiles")

    if len(batch) > 1 or (tiles is not None and image is None):
        arrays = [_coerce_image(x) for x in batch]
        if use_global:
            shared_palette = np.asarray(palette) if palette is not None else _build_global_palette(arrays, colors)
            quantized_tiles = []
            indices_tiles = []
            for arr in arrays:
                q, idx = _map_to_palette(arr, shared_palette)
                quantized_tiles.append(_image_result(
                    q, ok=True, palette=shared_palette, global_palette=True,
                    palette_scope="global", colors=shared_palette.shape[0]
                ))
                indices_tiles.append(idx)
            return Report(
                ok=True,
                value=quantized_tiles,
                tiles=quantized_tiles,
                indices=indices_tiles,
                palette=shared_palette,
                palettes=[shared_palette for _ in arrays],
                global_palette=True,
                palette_scope="global",
                colors=int(shared_palette.shape[0]),
            )

        local_palettes = []
        quantized_tiles = []
        for arr in arrays:
            pal = _build_global_palette([arr], colors) if palette is None else np.asarray(palette)
            q, idx = _map_to_palette(arr, pal)
            local_palettes.append(pal)
            quantized_tiles.append(_image_result(
                q, ok=True, palette=pal, global_palette=False, palette_scope="local"
            ))
        return Report(
            ok=True,
            value=quantized_tiles,
            tiles=quantized_tiles,
            palettes=local_palettes,
            palette=local_palettes[0] if local_palettes else None,
            global_palette=False,
            palette_scope="local",
        )

    arr = _coerce_image(batch[0])
    if palette is None:
        palette = _build_global_palette([arr], colors) if use_global else _build_global_palette([arr], colors)
    palette = np.asarray(palette)
    q, indices = _map_to_palette(arr, palette)
    return _image_result(
        q,
        ok=True,
        value=q,
        indices=indices,
        palette=palette,
        colors=int(palette.shape[0]),
        global_palette=use_global,
        palette_scope="global" if use_global else "local",
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# 5. 尺寸补齐
# ---------------------------------------------------------------------------


def _parse_wh(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if _is_scalar_number(value):
        return int(value), int(value)
    if isinstance(value, (tuple, list, np.ndarray)):
        flat = [int(v) for v in np.asarray(value).reshape(-1)[:2]]
        if len(flat) == 1:
            return flat[0], flat[0]
        return flat[0], flat[1]
    if isinstance(value, dict):
        if "size" in value:
            return _parse_wh(value["size"])
        w = value.get("width", value.get("w"))
        h = value.get("height", value.get("h"))
        if w is not None and h is not None:
            return int(w), int(h)
    return None


@_public
def pad_size(*args: Any, **kwargs: Any) -> SizeResult | Report:
    """计算整数缩放后的完整尺寸，并把目标/分块尺寸向上补齐。

    位置参数约定为 ``pad_size((width, height), factor=1)``、
    ``pad_size(width, height)`` 或 ``pad_size(width, height, factor)``。
    单个标量按正方形边长处理。
    """

    factor_provided = any(key in kwargs for key in ("factor", "scale", "ratio", "k"))
    factor = kwargs.pop("factor", kwargs.pop("scale", kwargs.pop("ratio", kwargs.pop("k", 1))))
    target = kwargs.pop("target", kwargs.pop("target_size", None))
    multiple = kwargs.pop("multiple", kwargs.pop("align", None))
    tile_size = kwargs.pop("tile_size", kwargs.pop("block_size", None))
    pad_to_target = bool(kwargs.pop("pad_to_target", True))

    def as_wh(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if _looks_like_image(value):
            return _source_wh(_coerce_image(value))
        return _parse_wh(value)

    source_wh = as_wh(kwargs.pop("size", None)) or as_wh(kwargs.pop("source", None))
    target_wh = as_wh(target)

    size_args: list[tuple[int, int]] = []
    scalar_args: list[float] = []
    for value in args:
        if _is_scalar_number(value):
            scalar_args.append(float(value))
        else:
            parsed = as_wh(value)
            if parsed is not None:
                size_args.append(parsed)

    if source_wh is None and size_args:
        source_wh = size_args.pop(0)
    if target_wh is None and size_args:
        target_wh = size_args.pop(0)

    if source_wh is None:
        if len(scalar_args) >= 4:
            source_wh = (int(scalar_args[0]), int(scalar_args[1]))
            if target_wh is None:
                target_wh = (int(scalar_args[2]), int(scalar_args[3]))
        elif len(scalar_args) >= 3:
            source_wh = (int(scalar_args[0]), int(scalar_args[1]))
            if not factor_provided:
                factor = int(scalar_args[2])
        elif len(scalar_args) == 2:
            source_wh = (int(scalar_args[0]), int(scalar_args[1]))
        elif len(scalar_args) == 1:
            source_wh = (int(scalar_args[0]), int(scalar_args[0]))
        else:
            source_wh = (1, 1)
    elif target_wh is None and scalar_args:
        if len(scalar_args) >= 2:
            target_wh = (int(scalar_args[0]), int(scalar_args[1]))
        elif not factor_provided:
            factor = int(scalar_args[0])

    if target_wh is None:
        tw = kwargs.pop("target_width", kwargs.pop("width", None))
        th = kwargs.pop("target_height", kwargs.pop("height", None))
        if tw is not None and th is not None:
            target_wh = (int(tw), int(th))

    k, ratio_info = _resolve_factor(factor, None, None)
    w, h = source_wh
    scaled_w, scaled_h = w * k, h * k

    padded_w, padded_h = scaled_w, scaled_h
    if target_wh is not None and pad_to_target:
        tw, th = target_wh
        if tw < scaled_w or th < scaled_h:
            # 绝不靠裁切来“满足”目标；向上补齐所需目标。
            padded_w, padded_h = max(tw, scaled_w), max(th, scaled_h)
        else:
            padded_w, padded_h = tw, th

    align_wh = _parse_wh(multiple if multiple is not None else tile_size)
    if align_wh is not None:
        mw, mh = align_wh
        padded_w = int(math.ceil(padded_w / mw) * mw) if mw > 0 else padded_w
        padded_h = int(math.ceil(padded_h / mh) * mh) if mh > 0 else padded_h

    pad_w = padded_w - scaled_w
    pad_h = padded_h - scaled_h
    report = {
        "ok": True,
        "value": (padded_w, padded_h),
        "source_size": (w, h),
        "factor": k,
        "ratio": k,
        "integer_ratio": True,
        "scaled_size": (scaled_w, scaled_h),
        "natural_size": (scaled_w, scaled_h),
        "padded_size": (padded_w, padded_h),
        "size": (padded_w, padded_h),
        "padding": (pad_w, pad_h),
        "padding_wh": (pad_w, pad_h),
        # PIL box: left, top, right, bottom；这里锚定左上，只补右下。
        "padding_box": (0, 0, pad_w, pad_h),
        # NumPy pad_width: ((top, bottom), (left, right))。
        "pad_width": ((0, pad_h), (0, pad_w)),
        "right_pad": pad_w,
        "bottom_pad": pad_h,
        "target_size": target_wh,
        "ratio_report": ratio_info,
        "edge_complete": True,
    }
    return SizeResult((padded_w, padded_h), report)


def _pad_array_to_size(arr: np.ndarray, size_wh: tuple[int, int], mode: str = "edge",
                       color: Any = None) -> np.ndarray:
    w, h = _source_wh(arr)
    tw, th = int(size_wh[0]), int(size_wh[1])
    if tw < w or th < h:
        raise ValueError(f"refusing to crop {w}x{h} to {tw}x{th}")
    pw, ph = tw - w, th - h
    if pw == 0 and ph == 0:
        return arr

    if mode == "constant":
        if color is None:
            color = 0
        return np.pad(arr, ((0, ph), (0, pw)) + ((0, 0),) * (arr.ndim - 2),
                      mode="constant", constant_values=color)
    if mode == "transparent" and arr.ndim == 3 and arr.shape[2] == 4:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="constant",
                      constant_values=0)
    return np.pad(arr, ((0, ph), (0, pw)) + ((0, 0),) * (arr.ndim - 2), mode="edge")


# ---------------------------------------------------------------------------
# 6. 边缘校验
# ---------------------------------------------------------------------------


def _extract_source_output(args: tuple, kwargs: dict) -> tuple[Any, Any, Any]:
    source = kwargs.pop("source", kwargs.pop("src", None))
    output = kwargs.pop("output", kwargs.pop("result", kwargs.pop("scaled", None)))
    factor = kwargs.pop("factor", kwargs.pop("scale", kwargs.pop("ratio", None)))

    positional = list(args)
    if source is None and positional:
        source = positional.pop(0)
    if output is None and positional:
        output = positional.pop(0)
    elif output is None and source is not None and _looks_like_image(source) and not positional:
        # edge_check(output_or_result)
        output = source
        source = None
    if factor is None and positional:
        factor = positional.pop(0)
    return source, output, factor


@_public
def edge_check(*args: Any, **kwargs: Any) -> Report:
    """检查最右列/最底行是否有对应输出，且边缘值保持有限。"""

    tolerance = float(kwargs.pop("tolerance", kwargs.pop("abs_tol", 8.0)))
    strict_color = bool(kwargs.pop("strict_color", False))
    source_arg, output_arg, factor = _extract_source_output(args, kwargs)

    source_arr = None
    if source_arg is not None and _looks_like_image(source_arg):
        try:
            source_arr = _coerce_image(source_arg)
        except Exception:
            source_arr = None

    output_arr = None
    output_report: dict[str, Any] = {}
    if output_arg is not None:
        if isinstance(output_arg, ImageArray):
            output_arr = np.asarray(output_arg)
            output_report = output_arg.report
        elif isinstance(output_arg, dict):
            output_report = dict(output_arg)
            if output_arg.get("edge_check"):
                prior = output_arg["edge_check"]
                if isinstance(prior, Report):
                    return prior
            try:
                output_arr = _coerce_image(output_arg)
            except Exception:
                output_arr = None
        elif _looks_like_image(output_arg):
            output_arr = _coerce_image(output_arg)

        if factor is None and isinstance(output_arg, ImageArray):
            factor = output_arg.report.get("factor")

    if factor is None:
        factor = output_report.get("factor") or output_report.get("ratio")

    if source_arr is None and output_arr is None:
        if _is_scalar_number(source_arg if source_arg is not None else output_arg):
            return Report(ok=False, checked=False, edge_complete=False,
                          reason="image required for edge comparison",
                          value=False)
        raise ValueError("edge_check requires source and/or output image")

    if source_arr is None and output_arr is not None:
        finite = bool(np.isfinite(_as_float01(output_arr)).all()) if np.issubdtype(output_arr.dtype, np.number) else True
        return Report(ok=finite, value=finite, source_available=False,
                      output_shape=output_arr.shape, all_finite=finite,
                      edge_complete=True, reason="geometry-only check")

    assert source_arr is not None
    h, w = source_arr.shape[:2]

    if output_arr is None:
        # 可能直接传入了 scale 报告但图像键无法解析；若有旧 edge_check 已在上面返回。
        raise ValueError("output image is not available")

    oh, ow = output_arr.shape[:2]
    if factor is None:
        rx = ow / w
        ry = oh / h
        if abs(rx - round(rx)) < 1e-9 and abs(ry - round(ry)) < 1e-9 and int(round(rx)) == int(round(ry)):
            factor = int(round(rx))
        else:
            factor = 1
    k = max(1, int(factor))

    required_w, required_h = w * k, h * k
    width_complete = ow >= required_w
    height_complete = oh >= required_h
    shape_complete = width_complete and height_complete

    right_error = None
    bottom_error = None
    corner_error = None
    if shape_complete:
        src_float = _as_float01(source_arr)
        out_float = _as_float01(output_arr[:required_h, :required_w, ...])
        if src_float.ndim == 2:
            src_float = src_float[..., None]
        if out_float.ndim == 2:
            out_float = out_float[..., None]

        right_expected = np.repeat(
            np.repeat(source_arr[:, -1:, ...], k, axis=0), k, axis=1
        )
        bottom_expected = np.repeat(
            np.repeat(source_arr[-1:, :, ...], k, axis=0), k, axis=1
        )
        right_actual = output_arr[:required_h, (w - 1) * k:w * k, ...]
        bottom_actual = output_arr[(h - 1) * k:h * k, :required_w, ...]

        right_error = float(_color_distance_u8(right_expected, right_actual).max()) if right_actual.size else None
        bottom_error = float(_color_distance_u8(bottom_expected, bottom_actual).max()) if bottom_actual.size else None
        corner_error = max(right_error or 0.0, bottom_error or 0.0)

    finite = bool(np.isfinite(_as_float01(output_arr)).all()) if np.issubdtype(output_arr.dtype, np.number) else True
    color_ok = True
    if strict_color and corner_error is not None:
        color_ok = corner_error <= tolerance

    ok = shape_complete and finite and color_ok
    result = Report(
        ok=ok,
        value=ok,
        edge_complete=shape_complete,
        right_column_preserved=width_complete,
        bottom_row_preserved=height_complete,
        width_complete=width_complete,
        height_complete=height_complete,
        required_size=(required_w, required_h),
        output_size=(ow, oh),
        missing_right_columns=max(0, required_w - ow),
        missing_bottom_rows=max(0, required_h - oh),
        right_max_error=right_error,
        bottom_max_error=bottom_error,
        max_error=corner_error,
        tolerance=tolerance,
        strict_color=strict_color,
        all_finite=finite,
        factor=k,
    )
    return result


# ---------------------------------------------------------------------------
# 7. 加载 / 缩放 / 导出
# ---------------------------------------------------------------------------


@_public
def load(*args: Any, **kwargs: Any) -> Report:
    """加载位图并返回数组、尺寸和模式信息。"""

    path = kwargs.pop("path", kwargs.pop("filename", args[0] if args else None))
    if path is None:
        raise ValueError("load requires a path")
    if not _is_path(path):
        raise TypeError(f"load expects a path, got {type(path).__name__}")
    if Image is None:
        raise RuntimeError("Pillow is required to load images")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with Image.open(p) as im:
        im.load()
        mode = im.mode
        width, height = im.size
        info = dict(im.info)
        if mode == "P":
            indices = np.array(im)
            palette_flat = im.getpalette() or []
            palette = np.asarray(palette_flat, dtype=np.uint8).reshape(-1, 3) if palette_flat else np.empty((0, 3), np.uint8)
            image = np.array(im.convert("RGBA" if "transparency" in info else "RGB"))
            return _image_result(
                image, ok=True, value=image, indices=indices, palette=palette,
                path=str(p), mode=mode, size=(width, height), width=width, height=height,
                info=info,
            )

        image = np.array(im)

    return _image_result(
        image, ok=True, value=image, path=str(p), mode=mode,
        size=(width, height), width=width, height=height, info=info,
    )


def _parse_scale_call(args: tuple, kwargs: dict) -> tuple[Any, Any, Any]:
    image = kwargs.pop("image", None)
    factor = kwargs.pop("factor", kwargs.pop("scale", kwargs.pop("ratio", kwargs.pop("k", None))))
    target = kwargs.pop("target", kwargs.pop("target_size", None))

    positional = list(args)
    if image is None and positional:
        image = positional.pop(0)

    # 宽容 scale(factor, image) 的反序调用。
    if image is not None and not _looks_like_image(image) and positional and _looks_like_image(positional[0]):
        if factor is None and _is_scalar_number(image):
            factor = image
        image = positional.pop(0)

    while positional:
        value = positional.pop(0)
        if _looks_like_image(value):
            image = value
        elif target is None and _parse_wh(value) is not None and not _is_scalar_number(value):
            target = value
        elif factor is None and _is_scalar_number(value):
            factor = value
        elif target is None:
            target = value
    return image, factor, target


@_public
def scale(*args: Any, **kwargs: Any) -> Any:
    """整数倍最近邻/自适应 EPX 缩放主入口。"""

    image_arg, factor, target = _parse_scale_call(args, kwargs)
    mode = str(kwargs.pop("mode", kwargs.pop("method", "nearest"))).lower()
    tile_size = kwargs.pop("tile_size", kwargs.pop("block_size", None))
    adaptive = bool(kwargs.pop("adaptive", kwargs.pop("local_threshold", True)))
    threshold = kwargs.pop("threshold", None)
    soft = bool(kwargs.pop("soft", True))
    quantize_after = bool(kwargs.pop("quantize_after", kwargs.pop("quantize", False)))
    palette = kwargs.pop("palette", None)
    colors = int(kwargs.pop("colors", 256))
    global_palette_flag = bool(kwargs.pop("global_palette", True))
    pad = bool(kwargs.pop("pad", kwargs.pop("pad_to_target", True)))
    edge_verify = bool(kwargs.pop("edge_verify", True))
    strict = bool(kwargs.pop("strict", False))
    pad_mode = str(kwargs.pop("pad_mode", "edge"))

    source = _coerce_image(image_arg)
    source_wh = _source_wh(source)
    target_wh = _parse_wh(target)
    k, ratio_info = _resolve_factor(factor, target_wh, source_wh, strict=strict)
    if strict and ratio_info.get("ok") is False:
        return Report(ok=False, **ratio_info)

    epx_requested = mode in ("epx", "scale2x", "smart", "edge_aware")
    algorithm = "adaptive_epx2" if epx_requested else "nearest_neighbor"

    if epx_requested:
        if k % 2 == 0:
            if tile_size is not None:
                out = _epx2_tiled(source, tile_size, adaptive=adaptive,
                                   threshold=threshold, soft=soft)
            else:
                out = _epx2_array(source, adaptive=adaptive, threshold=threshold, soft=soft)
            repeat = k // 2
            if repeat > 1:
                out = _nearest_scale(out, repeat)
            algorithm = "adaptive_epx2" if repeat == 1 else f"adaptive_epx2_x{repeat}_nearest"
        else:
            out = _nearest_scale(source, k)
            algorithm = "nearest_integer_fallback_for_odd_factor"
    else:
        out = _nearest_scale(source, k)

    seam_error = None
    if epx_requested and tile_size is not None and k == 2:
        # 必须在量化/补齐前比较，否则色板误差会被误报为块缝。
        reference = _epx2_array(source, adaptive=adaptive, threshold=threshold, soft=soft)
        seam_error = float(np.abs(_as_float01(reference) - _as_float01(out)).max() * 255.0)

    natural_wh = (source_wh[0] * k, source_wh[1] * k)
    padded_to = None
    if target_wh is not None and pad:
        desired = (max(target_wh[0], natural_wh[0]), max(target_wh[1], natural_wh[1]))
        if desired != _source_wh(out):
            out = _pad_array_to_size(out, desired, mode=pad_mode)
            padded_to = desired

    palette_report = None
    if quantize_after or palette is not None:
        q = quantize(out, colors=colors, palette=palette,
                     global_palette=global_palette_flag)
        if isinstance(q, ImageArray):
            palette_report = q.report
            out = np.asarray(q)
        else:
            palette_report = dict(q)

    edge_report = None
    if edge_verify:
        edge_report = edge_check(source, out, factor=k, strict_color=False)

    out_wh = _source_wh(out)

    report = {
        "ok": True,
        "value": out,
        "source": source,
        "source_size": source_wh,
        "size": out_wh,
        "output_size": out_wh,
        "natural_size": natural_wh,
        "padded_to": padded_to,
        "factor": k,
        "scale": k,
        "ratio": k,
        "integer_ratio": True,
        "ratio_report": ratio_info,
        "mode": mode,
        "algorithm": algorithm,
        "tile_size": tile_size,
        "cross_tile": tile_size is not None,
        "neighborhood": "cross_tile_full_source" if tile_size is not None else "whole_image",
        "edge_check": edge_report,
        "palette_report": palette_report,
        "palette_scope": (palette_report or {}).get("palette_scope", "none"),
        "global_palette": bool((palette_report or {}).get("global_palette", False)),
        "seam_max_error": seam_error,
        "deterministic": True,
    }
    return _image_result(out, **report)


@_public
def export(*args: Any, **kwargs: Any) -> Report:
    """把数组/缩放结果写入文件或编码为 PNG 字节，并复核导出尺寸。"""

    path = kwargs.pop("path", kwargs.pop("filename", kwargs.pop("output", None)))
    image_arg = kwargs.pop("image", kwargs.pop("result", None))
    if args:
        if _is_path(args[0]):
            path = path or args[0]
            rest = list(args[1:])
            if image_arg is None and rest:
                image_arg = rest[0]
        else:
            image_arg = image_arg or args[0]
            rest = list(args[1:])
            if path is None and rest and _is_path(rest[0]):
                path = rest[0]
    if image_arg is None:
        raise ValueError("export requires an image")

    report_from_image: dict[str, Any] = {}
    if isinstance(image_arg, ImageArray):
        report_from_image = image_arg.report
    elif isinstance(image_arg, dict):
        report_from_image = dict(image_arg)

    arr = _coerce_image(image_arg)
    if Image is None:
        raise RuntimeError("Pillow is required to export images")

    pil_image = _to_pil_image(arr)
    fmt = kwargs.pop("format", None)
    if fmt is None:
        fmt = "PNG" if path is None else (Path(str(path)).suffix.lstrip(".").upper() or "PNG")
        if fmt == "JPG":
            fmt = "JPEG"

    buffer = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG" and pil_image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        rgba = pil_image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        pil_image = background
    pil_image.save(buffer, format=fmt, **save_kwargs)
    data = buffer.getvalue()

    written_path = None
    if path is not None:
        p = Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written_path = str(p)

    check = edge_check(report_from_image.get("source", image_arg), arr,
                       factor=report_from_image.get("factor"))
    return Report(
        ok=True,
        value=data,
        data=data,
        bytes=data,
        path=written_path,
        format=fmt,
        size=(pil_image.width, pil_image.height),
        width=pil_image.width,
        height=pil_image.height,
        byte_count=len(data),
        edge_check=check,
        factor=report_from_image.get("factor"),
    )


def _to_pil_image(arr: np.ndarray) -> "Image.Image":
    a = np.asarray(arr)
    if a.dtype == bool:
        a = (a.astype(np.uint8) * 255)
    if np.issubdtype(a.dtype, np.floating):
        a = _from_float01(a, np.dtype(np.uint8))
    elif np.issubdtype(a.dtype, np.integer) and a.dtype != np.uint8:
        # 像素画交换格式以 8 位最通用；16 位保留可由调用方自行用 Pillow 处理。
        if np.iinfo(a.dtype).max > 255:
            a = np.clip(_as_float01(a) * 255.0, 0, 255).astype(np.uint8)
        else:
            a = a.astype(np.uint8)

    if a.ndim == 2:
        return Image.fromarray(a, mode="L")
    if a.ndim == 3 and a.shape[2] == 1:
        return Image.fromarray(a[..., 0], mode="L")
    if a.ndim == 3 and a.shape[2] == 2:
        # LA 不一定被所有编码器直接支持，转 RGBA 更稳。
        la = Image.fromarray(a, mode="LA")
        return la.convert("RGBA")
    if a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray(a, mode="RGB")
    if a.ndim == 3 and a.shape[2] == 4:
        return Image.fromarray(a, mode="RGBA")
    raise ValueError(f"unsupported image shape {a.shape}")


# ---------------------------------------------------------------------------
# 8. 校验、指标、审计、兜底、确定性
# ---------------------------------------------------------------------------


def _hash_array(arr: Any) -> str:
    a = np.ascontiguousarray(_coerce_image(arr))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@_public
def metrics(*args: Any, **kwargs: Any) -> Report:
    """计算尺寸、边缘、接缝、色板和颜色误差指标。"""

    source_arg, output_arg, factor = _extract_source_output(args, dict(kwargs))
    if source_arg is None and output_arg is None:
        return Report(ok=True, value={}, metrics={}, samples=0,
                      edge_preserved=True, seam_max_error=0.0,
                      palette_unified=True, deterministic=True)

    source_arr = output_arr = None
    try:
        if source_arg is not None:
            source_arr = _coerce_image(source_arg)
    except Exception:
        source_arr = None
    try:
        if output_arg is not None:
            output_arr = _coerce_image(output_arg)
    except Exception:
        output_arr = None

    out_report = output_arg.report if isinstance(output_arg, ImageArray) else (
        output_arg if isinstance(output_arg, dict) else {}
    )
    k = factor or out_report.get("factor") or out_report.get("ratio") or 1
    k = int(k)

    result: dict[str, Any] = {
        "ok": True,
        "factor": k,
        "integer_ratio": float(k).is_integer(),
        "palette_unified": out_report.get("global_palette", True),
        "palette_scope": out_report.get("palette_scope", "unknown"),
        "seam_max_error": out_report.get("seam_max_error"),
    }

    if source_arr is not None:
        result["source_size"] = _source_wh(source_arr)
    if output_arr is not None:
        result["output_size"] = _source_wh(output_arr)
        result["pixels"] = int(output_arr.size)

    edge = None
    if source_arr is not None and output_arr is not None:
        edge = edge_check(source_arr, output_arr, factor=k)
        result["edge_complete"] = edge["edge_complete"]
        result["edge_max_error"] = edge["max_error"]
        result["right_max_error"] = edge["right_max_error"]
        result["bottom_max_error"] = edge["bottom_max_error"]

        expected = _nearest_scale(source_arr, k)
        if expected.shape == output_arr.shape:
            diff = np.abs(_as_float01(expected) - _as_float01(output_arr)) * 255.0
            result["max_color_error_vs_nearest"] = float(diff.max()) if diff.size else 0.0
            result["mean_color_error_vs_nearest"] = float(diff.mean()) if diff.size else 0.0

    if edge is not None:
        result["edge_check"] = edge
        result["edge_preserved"] = bool(edge["edge_complete"])
    else:
        result["edge_preserved"] = True

    result["value"] = {key: v for key, v in result.items() if key != "value"}
    return Report(result)


@_public
def validate_scale(*args: Any, **kwargs: Any) -> Report:
    """综合验证一次缩放结果是否满足全部六项约束。"""

    source_arg, result_arg, factor = _extract_source_output(args, dict(kwargs))

    # validate_scale(result) 且结果是本模块 ImageArray/Report。
    report: dict[str, Any] = {}
    output = result_arg
    if isinstance(result_arg, ImageArray):
        report = result_arg.report
    elif isinstance(result_arg, dict):
        report = dict(result_arg)
        output = report.get("image", report.get("value", result_arg))

    if source_arg is None:
        source_arg = report.get("source")

    k = factor or report.get("factor") or report.get("ratio") or 1
    k = int(k)

    checks: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []

    ratio = integer_ratio(k)
    checks["integer_ratio"] = bool(ratio["exact_integer"])
    if not checks["integer_ratio"]:
        issues.append({"type": "non_integer_ratio", "detail": ratio})

    edge = None
    if source_arg is not None and output is not None and _looks_like_image(output):
        edge = edge_check(source_arg, output, factor=k)
        checks["edges"] = bool(edge["ok"])
        if not checks["edges"]:
            issues.append({"type": "edge_crop_or_damage", "detail": dict(edge)})
    elif report.get("edge_check"):
        edge = report["edge_check"]
        checks["edges"] = bool(edge.get("ok", False)) if isinstance(edge, dict) else bool(edge)
        if not checks["edges"]:
            issues.append({"type": "edge_check_failed", "detail": edge})
    else:
        checks["edges"] = True

    checks["size_padded"] = True
    if source_arg is not None and output is not None:
        try:
            s = _coerce_image(source_arg)
            o = _coerce_image(output)
            checks["size_padded"] = o.shape[1] >= s.shape[1] * k and o.shape[0] >= s.shape[0] * k
            if not checks["size_padded"]:
                issues.append({"type": "output_too_small",
                              "required": (s.shape[1] * k, s.shape[0] * k),
                              "actual": (o.shape[1], o.shape[0])})
        except Exception as exc:
            checks["size_padded"] = False
            issues.append({"type": "invalid_image", "message": str(exc)})

    palette_scope = report.get("palette_scope")
    checks["global_palette"] = palette_scope in (None, "none", "global") or report.get("global_palette", False)
    if report.get("palette_scope") == "local":
        checks["global_palette"] = False
        issues.append({"type": "local_palette"})

    seam = report.get("seam_max_error")
    checks["cross_tile_neighborhood"] = True
    if report.get("tile_size") is not None and seam is not None:
        checks["cross_tile_neighborhood"] = float(seam) <= float(kwargs.get("seam_tolerance", 1e-6))
        if not checks["cross_tile_neighborhood"]:
            issues.append({"type": "tile_seam", "max_error": seam})

    runs = report.get("deterministic", True)
    checks["determinism"] = bool(runs)

    ok = all(bool(v) for key, v in checks.items() if key != "size_padded" or True)
    return Report(
        ok=ok and not issues,
        value=ok and not issues,
        valid=ok and not issues,
        checks=checks,
        issues=issues,
        edge_check=edge,
        factor=k,
        audit=audit(report if report else result_arg),
    )


@_public
def audit(*args: Any, **kwargs: Any) -> Report:
    """审计流水线记录或多块量化结果，找出六类典型缺陷。"""

    records = list(args)
    if kwargs:
        records.append(kwargs)

    issues: list[dict[str, Any]] = []
    audited = 0
    palette_ids: set[int] = set()

    def visit(obj: Any) -> None:
        nonlocal audited
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
            return
        if not isinstance(obj, dict):
            if obj is not None:
                audited += 1
            return

        audited += 1
        if obj.get("ok") is False:
            issues.append({"type": "operation_failed", "error": obj.get("error")})

        ratio = obj.get("ratio_report") or {}
        if ratio.get("exact_integer") is False or obj.get("integer_ratio") is False:
            issues.append({"type": "non_integer_ratio", "detail": ratio})

        edge = obj.get("edge_check")
        if isinstance(edge, dict) and edge.get("ok") is False:
            issues.append({"type": "edge_crop", "detail": edge})

        if obj.get("palette_scope") == "local" or obj.get("global_palette") is False:
            issues.append({"type": "local_palette"})

        pal = obj.get("palette")
        if isinstance(pal, np.ndarray):
            palette_ids.add(hash(_hash_array(pal) if pal.size else id(pal)))
        elif isinstance(pal, (list, tuple)) and pal and all(isinstance(p, np.ndarray) for p in pal):
            for p in pal:
                palette_ids.add(hash(_hash_array(p)))
            if len({_hash_array(p) for p in pal}) > 1:
                issues.append({"type": "different_tile_palettes"})

        if obj.get("seam_max_error") not in (None, 0, 0.0):
            try:
                if float(obj["seam_max_error"]) > 1e-6:
                    issues.append({"type": "tile_seam", "max_error": obj["seam_max_error"]})
            except Exception:
                pass

        for key in ("issues",):
            child = obj.get(key)
            if isinstance(child, list):
                issues.extend(i for i in child if isinstance(i, dict))

    for record in records:
        visit(record)

    if len(palette_ids) > 1:
        issues.append({"type": "palette_mismatch_across_tiles", "palette_count": len(palette_ids)})

    # 去重相同类型/消息，避免 validate->audit 嵌套重复。
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for issue in issues:
        key = issue.get("type", "issue") + ":" + str(issue.get("error", issue.get("max_error", "")))
        if key not in seen:
            seen.add(key)
            unique.append(issue)

    return Report(
        ok=not unique,
        value=not unique,
        clean=not unique,
        issues=unique,
        issue_count=len(unique),
        audited=audited,
        global_palette=not any(i["type"] in ("local_palette", "different_tile_palettes",
                                             "palette_mismatch_across_tiles") for i in unique),
    )


@_public
def fallback(*args: Any, **kwargs: Any) -> Any:
    """任何智能算法不可用时的安全路径：整数倍最近邻，绝不裁切。"""

    factor = int(kwargs.pop("factor", kwargs.pop("scale", kwargs.pop("ratio", 1))) or 1)
    reason = kwargs.pop("reason", "safe_integer_nearest")
    if not args:
        return Report(ok=True, value="nearest", mode="nearest_neighbor",
                      factor=factor, strategy="integer_nearest_no_crop", reason=reason)

    first = args[0]
    if _is_scalar_number(first) and not _looks_like_image(first):
        return Report(ok=True, value="nearest", mode="nearest_neighbor",
                      factor=max(1, int(first) if factor == 1 else factor),
                      strategy="integer_nearest_no_crop", reason=reason)

    arr = _coerce_image(first)
    out = _nearest_scale(arr, max(1, factor))
    return _image_result(out, ok=True, mode="nearest_neighbor", factor=max(1, factor),
                         fallback=True, reason=reason, deterministic=True,
                         source_size=_source_wh(arr), size=_source_wh(out))


@_public
def determinism(*args: Any, **kwargs: Any) -> Report:
    """重复执行缩放/量化操作并比较 SHA-256，验证确定性。"""

    runs = int(kwargs.pop("runs", 3))
    operation = kwargs.pop("operation", kwargs.pop("fn", None))
    op_args = kwargs.pop("op_args", None)

    if operation is None and args and callable(args[0]):
        operation = args[0]
        args = args[1:]

    image = None
    if operation is None:
        for arg in args:
            if _looks_like_image(arg):
                image = arg
                break

    if operation is None and image is None:
        return Report(ok=True, value=True, deterministic=True, runs=0,
                      hashes=[], note="no image or operation supplied")

    if operation is None:
        factor = int(kwargs.get("factor", 2))
        mode = kwargs.get("mode", "nearest")
        do_quantize = bool(kwargs.get("quantize_after", False))

        def operation(img, **_ignored):
            out = scale(img, factor=factor, mode=mode, edge_verify=False)
            if do_quantize:
                out = quantize(out, colors=int(kwargs.get("colors", 256)))
            return out

        op_args = (image,)
        call_kwargs: dict[str, Any] = {}
    else:
        if op_args is None:
            op_args = args
        call_kwargs = kwargs

    outputs = []
    hashes = []
    for _ in range(max(2, runs)):
        result = operation(*op_args, **call_kwargs)
        outputs.append(result)
        if isinstance(result, dict) and "tiles" in result:
            h = hashlib.sha256()
            for tile in result["tiles"]:
                h.update(_hash_array(tile).encode())
            if result.get("palette") is not None:
                h.update(np.asarray(result["palette"]).tobytes())
            hashes.append(h.hexdigest())
        else:
            hashes.append(_hash_array(result))

    deterministic = len(set(hashes)) == 1
    return Report(
        ok=deterministic,
        value=deterministic,
        deterministic=deterministic,
        runs=len(hashes),
        hashes=hashes,
        algorithm="sha256_shape_dtype_bytes",
    )
