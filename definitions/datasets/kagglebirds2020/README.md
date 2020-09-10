Preparation
-----------

For training, it is strongly advisable to precompute .wav files for all the
training clips and store them somewhere locally. The `collect_and_convert.sh`
script helps with this. For example, you could do:

```bash
ln -s /mnt/data/shared/birds_kaggle_2020_wav audio
if [ ! -e audio ]; then
mkdir -p /mnt/data/shared/birds_kaggle_2020_wav
./collect_and_convert.sh audio/train/official /share/cp/datasets/birds_kaggle_2020/official/train_audio/
./collect_and_convert.sh audio/train/xeno-canto /share/cp/datasets/birds_kaggle_2020/external/xeno-canto/
fi
```

The first command creates a symlink to some directory on a local disk, called
`audio` for convenience, the next two put some of the training files there,
converted to 22.05 kHz .wav files. The dataset reader does not care about the
directory structure, but requires all files to have a different base name.

For more portability, the training csv files can also be symlinked to this
directory instead of giving their full path in the configuration:
```bash
ln -s /share/cp/datasets/birds_kaggle_2020/official/train.csv
ln -s /share/cp/datasets/birds_kaggle_2020/external/xeno-canto/train_extended.csv
```

For mixing in background noise, the .wav files need to be precomputed as well:
```bash
if [ ! -e audio/noise ]; then
./collect_and_convert.sh audio/noise/chernobyl /share/cp/datasets/birds_kaggle_2020/external/chernobyl
./collect_and_convert.sh audio/noise/birdvox-full-night /share/cp/datasets/birds_kaggle_2020/external/birdvox-full-night
fi
```
In addition, the .csv files need to be named and placed correctly:
```bash
for fn in /share/cp/datasets/birds_kaggle_2020/external/chernobyl/*.txt; do bn="${fn##*/}"; cp -a "$fn" audio/noise/chernobyl/"${bn%%.*}.csv"; done
for fn in /share/cp/datasets/birds_kaggle_2020/external/birdvox-full-night/*.csv; do bn="${fn##*/}"; cp -a "$fn" audio/noise/birdvox-full-night/"${bn/csv-annotations/flac-audio}"; done
```
