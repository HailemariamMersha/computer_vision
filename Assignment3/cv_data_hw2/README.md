## Data Organization

All data are stored in the `data/` folder, the organization is as follows:

```
data/
├── Pair_S_000001_3503_3563/
│   ├── S_000001_frame_3503.png
│   ├── S_000001_frame_3503.annotation.txt
│   ├── S_000001_frame_3563.png
│   ├── S_000001_frame_3563.annotation.txt
├── Pair_S_000001_3563_3623/
│   ├── S_000001_frame_3563.png
│   ├── S_000001_frame_3563.annotation.txt
│   ├── S_000001_frame_3563.png
│   ├── S_000001_frame_3623.annotation.txt
```

It is guaranteed that each subdirectory in the data folder contains at least 4 files: a pair of PNG image files and a pair of annotation txt files. Each subdirectory may contain more than 4 files (such as the bounding box images), but we don't have to worry about those.

It is also guaranteed that the images and annotations are named as `*_aaaa.png`, `*_aaaa.annotation.txt`, `*_bbbb.png`, and `*_bbbb.annotation.txt`. The `*` represent 0 or more of any character; `aaaa` and `bbbb` represent two different 4-digit frame numbers.

It is guaranteed that the labels are in the correct format (contains 8 space-separated numbers).

It has NOT be verified whether the PNGs have the same dimension, so a transform pipeline before training is necessary. It is also NOT verified whether the annotations are correct. The hope is that a few incorrect annotations won't affect the training result too much.

## Data Index

A file named `index.txt` in provided in the root directory of this repository. Each row of the file contains 4 comma-separated values: 4 relative paths to a pair of images and their annotations. In the following order:

```
[path to frame 1], [path to annotation 1], [path to frame 2], [path to annotation 2]
```

Here is an example of what the file looks like:

```
data/Pair_S_000001_3503_3563/S_000001_frame_3503.png,data/Pair_S_000001_3503_3563/S_000001_frame_3503.annotation.txt,data/Pair_S_000001_3503_3563/S_000001_frame_3563.png,data/Pair_S_000001_3503_3563/S_000001_frame_3563.annotation.txt
data/Pair_S_000001_3563_3623/S_000001_frame_3563.png,data/Pair_S_000001_3563_3623/S_000001_frame_3563.annotation.txt,data/Pair_S_000001_3563_3623/S_000001_frame_3623.png,data/Pair_S_000001_3563_3623/S_000001_frame_3623.annotation.txt
```

Iterating through this file line-by-line should simplify the data loading process.
