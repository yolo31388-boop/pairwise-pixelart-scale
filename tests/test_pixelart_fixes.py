"""六项像素画缩放缺陷的回归测试。

原 ``test_pixelart_scale.py`` 只验证公开入口可调用；本文件验证题目描述中的
整数倍率、局部阈值、跨块邻域、全局色板、尺寸补齐和边缘校验。
"""

from __future__ import annotations

import numpy as np

import imaging.pixelart_scale as mod


def test_integer_ratio_rejects_fractional_request() -> None:
    assert int(mod.integer_ratio(4)) == 4
    exact = mod.integer_ratio((15, 17), (60, 68))
    assert exact["factor"] == 4
    assert exact["exact_integer"] is True

    fractional = mod.integer_ratio(2.5)
    assert fractional["exact_integer"] is False
    assert float(fractional) in (2.0, 3.0)

    strict = mod.integer_ratio(2.5, strict=True)
    assert strict["ok"] is False


def test_local_contrast_threshold_is_adaptive() -> None:
    flat = np.full((3, 3), 128, dtype=np.uint8)
    contrast = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)

    low = mod.threshold_local_contrast(flat)["threshold"]
    high = mod.threshold_local_contrast(contrast)["threshold"]
    assert high > low

    result = mod.threshold_local_contrast(contrast)
    assert result["threshold_map"].shape == contrast.shape
    assert np.all(np.isfinite(result["threshold_map"]))


def test_cross_tile_neighborhood_contains_outside_rows() -> None:
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    result = mod.neighborhood_cross_tile(image, (2, 4, 2, 4), radius=1)

    assert result["cross_tile"] is True
    # 2x2 tile plus one source pixel on every side.
    assert result["neighborhood"].shape == (4, 4)
    neighborhood = result["neighborhood"]
    assert neighborhood[1, 1] == image[2, 2]
    assert neighborhood[0, 0] == image[1, 1]
    assert neighborhood[-1, -1] == image[4, 4]


def test_tiled_epx_matches_whole_image_epx() -> None:
    rng = np.random.default_rng(7)
    # 用块状/阶梯图案而非纯噪声，避免偶然对称。
    source = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    source[2:5, 3:8] = np.array([240, 30, 30], dtype=np.uint8)
    source[5:8, 1:4] = np.array([30, 80, 240], dtype=np.uint8)

    whole = mod.scale(source, factor=2, mode="epx", edge_verify=False)
    tiled = mod.scale(source, factor=2, mode="epx", tile_size=3, edge_verify=False)

    assert tiled.shape == whole.shape
    assert np.array_equal(np.asarray(tiled), np.asarray(whole))
    assert tiled["seam_max_error"] == 0.0


def test_global_palette_is_shared_between_tiles() -> None:
    red = np.full((3, 3, 3), (220, 20, 20), dtype=np.uint8)
    blue = np.full((3, 3, 3), (20, 40, 220), dtype=np.uint8)

    palette = mod.global_palette([red, blue], colors=2)
    assert palette["global_palette"] is True
    assert palette["palette"].shape[0] == 2

    result = mod.quantize([red, blue], colors=2, global_palette=True)
    assert result["palette_scope"] == "global"
    assert np.array_equal(result["palettes"][0], result["palettes"][1])
    assert result["tiles"][0].shape == red.shape


def test_odd_width_is_not_cropped(tmp_path) -> None:
    source = np.zeros((5, 7, 3), dtype=np.uint8)
    source[:] = (40, 160, 80)
    source[:, -1] = (220, 40, 40)

    size = mod.pad_size((7, 5), factor=3)
    assert tuple(size) == (21, 15)
    assert size["padded_size"] == (21, 15)

    result = mod.scale(source, factor=3)
    assert result.shape == (15, 21, 3)
    assert result["edge_check"]["right_column_preserved"] is True
    assert result["edge_check"]["bottom_row_preserved"] is True

    path = tmp_path / "odd.png"
    exported = mod.export(result, path)
    assert exported["ok"] is True
    loaded = mod.load(path)
    assert loaded["size"] == (21, 15)
    assert loaded["image"].shape[1] == 21


def test_pad_to_target_adds_right_and_bottom_pixels() -> None:
    source = np.full((3, 5, 3), 123, dtype=np.uint8)
    result = mod.scale(source, factor=2, target=(20, 14), pad_mode="edge")

    assert result.shape == (14, 20, 3)
    assert result["natural_size"] == (10, 6)
    assert result["padded_to"] == (20, 14)
    assert result["edge_check"]["ok"] is True


def test_edge_check_detects_missing_right_column() -> None:
    source = np.arange(21, dtype=np.uint8).reshape(3, 7)
    scaled = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    cropped = scaled[:, :-1]

    report = mod.edge_check(source, cropped, factor=2)
    assert report["ok"] is False
    assert report["right_column_preserved"] is False
    assert report["missing_right_columns"] == 1


def test_validate_and_determinism_pass() -> None:
    source = np.zeros((6, 6, 3), dtype=np.uint8)
    source[1:4, 1:4] = (255, 255, 255)
    result = mod.scale(source, factor=2, mode="epx", tile_size=2,
                       quantize_after=True, colors=4)

    validation = mod.validate_scale(result)
    assert validation["ok"] is True, validation["issues"]

    first = mod.scale(source, factor=2, mode="epx")
    second = mod.scale(source, factor=2, mode="epx")
    assert np.array_equal(np.asarray(first), np.asarray(second))

    deterministic = mod.determinism(source, factor=2, mode="epx", runs=3)
    assert deterministic["deterministic"] is True
    assert len(set(deterministic["hashes"])) == 1
