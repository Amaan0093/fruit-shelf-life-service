import pandas as pd

from data.dataset_loader import source_photo, stratified_split

COPIES = ["", "rotated_by_15_", "translation_", "vertical_flip_", "saltandpepper_"]


def make_df(n_photos=40):
    paths = [
        f"dataset/apple/stage_1/train_{copy}Photo {i}.png"
        for i in range(n_photos)
        for copy in COPIES
    ]
    return pd.DataFrame({"path": paths, "fruit": "apple", "stage": 1, "days": 14.0})


def test_source_photo_removes_copy_prefixes():
    assert source_photo(r"dataset\\apple\\test_rotated_by_60_Shot 1.png") == "Shot 1.png"
    assert source_photo("dataset/apple/train_vertical_flip_Shot 1.png") == "Shot 1.png"
    assert source_photo("dataset/apple/test_Shot 1.png") == "Shot 1.png"


def test_no_photo_appears_in_two_splits():
    train, val, test = stratified_split(make_df())
    groups = [set(part["path"].map(source_photo)) for part in (train, val, test)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_split_keeps_every_image():
    df = make_df()
    assert sum(len(part) for part in stratified_split(df)) == len(df)