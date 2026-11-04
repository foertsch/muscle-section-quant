import numpy as np
import pytest

from msq.core.objects import joint_counts, region_rows, upsample_labels, zone_mean_sd


def two_squares():
    labels = np.zeros((12, 16), np.uint32)
    labels[0:4, 2:6] = 1  # touches the top edge
    labels[5:10, 8:14] = 3  # label 2 is missing
    return labels


def test_region_rows_in_label_order_with_full_resolution_positions():
    df = region_rows(two_squares(), px_um=2.0, f=4)
    assert df["label"].tolist() == [1, 3]
    assert list(df.columns) == ["label", "area_um2", "min_feret_um", "max_feret_um", "solidity", "eccentricity",
                                "centroid_y", "centroid_x", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
                                "touches_border"]
    one, three = df.iloc[0], df.iloc[1]
    assert one.area_um2 == 16 * 4.0 and three.area_um2 == 30 * 4.0 and one.solidity == 1.0
    assert (one.centroid_y, one.centroid_x) == ((1.5 + 0.5) * 4, (3.5 + 0.5) * 4)  # full-resolution pixels
    assert (three.bbox_y0, three.bbox_x0, three.bbox_y1, three.bbox_x1) == (20, 32, 40, 56)
    assert one.min_feret_um == 8.0 and three.max_feret_um == pytest.approx(2.0 * np.hypot(5, 6))
    assert df["touches_border"].tolist() == [True, False]


def test_region_rows_adds_the_extra_columns_after_the_shared_ones():
    df = region_rows(two_squares(), 1.0, 1, lambda rp: {"first_row": int(rp.bbox[0])})
    assert list(df.columns)[-2:] == ["touches_border", "first_row"] and df["first_row"].tolist() == [0, 5]
    assert region_rows(np.zeros((5, 5), np.uint32), 1.0, 1).empty


def test_joint_counts_by_label_and_value():
    zone = np.array([[0, 1, 1], [2, 2, 2]])
    values = np.array([[3, 0, 1], [1, 1, 3]])
    got = joint_counts(zone, values, 4, 4)
    assert got.dtype == np.int64 and got.shape == (4, 4)
    assert got.tolist() == [[0, 0, 0, 1], [1, 1, 0, 0], [0, 2, 0, 1], [0, 0, 0, 0]]


def test_zone_mean_sd_per_label():
    zone = np.array([[0, 1, 1], [2, 2, 2]])
    img = np.array([[9.0, 2.0, 4.0], [100.0, 100.0, 100.0]])
    cnt, mean, sd = zone_mean_sd(zone, img, 3)
    assert cnt.tolist() == [1, 2, 3, 0] and mean[1] == 3.0 and sd[1] == 1.0 and sd[2] == 0 and np.isnan(mean[3])


def test_upsample_labels_repeats_crops_and_pads():
    lab = np.arange(6, dtype=np.uint32).reshape(2, 3)
    full = np.repeat(np.repeat(lab, 3, 0), 3, 1)
    assert np.array_equal(upsample_labels(lab, 3), full)
    assert np.array_equal(upsample_labels(lab, 3, (5, 8)), full[:5, :8])  # cropped
    padded = upsample_labels(lab, 3, (8, 10))
    assert np.array_equal(padded[:6, :9], full) and not padded[6:].any() and not padded[:, 9:].any()
    assert upsample_labels(lab > 2, 3, (6, 9)).dtype == bool


def test_upsample_labels_window_and_clip():
    lab = np.arange(20, dtype=np.uint32).reshape(4, 5)
    full = np.repeat(np.repeat(lab, 4, 0), 4, 1)
    assert np.array_equal(upsample_labels(lab, 4, (7, 9), (5, 6)), full[5:12, 6:15])
    past = upsample_labels(lab, 4, (6, 6), (13, 17), clip=True)  # rows 13..18 and columns 17..22 of a 16 x 20 image
    assert np.array_equal(past[:3, :3], full[13:16, 17:20]) and (past[3:, :3] == full[15, 17:20]).all()
    assert (past[:, 3:] == past[:, 2:3]).all()  # the edge value, not zeros
    assert not upsample_labels(lab, 4, (6, 6), (13, 17))[3:].any()  # without clip: zeros
