Preparation
-----------

For training, it is strongly advisable to precompute .wav files for all the
training clips and store them somewhere locally. The `collect_and_convert.sh`
script helps with this. For example, you could do:

```bash
mkdir -p ~/.localhost/data/kaggle_birds_wav
ln -s ~/.localhost/data/kaggle_birds_wav audio
./collect_and_convert.sh audio/train/official /share/cp/datasets/birds_kaggle_2020/official/train_audio/
./collect_and_convert.sh audio/train/xeno-canto /share/cp/datasets/birds_kaggle_2020/external/xeno-canto/
```

The first command creates a symlink to some directory on a local disk, called
`audio` for convenience, the next two put some of the training files there,
converted to 22.5 kHz .wav files. The dataset reader does not care about the
directory structure, but requires all files to have a different base name.

For more portability, the training csv files can also be symlinked to this
directory instead of giving their full path in the configuration:
```bash
ln -s /share/cp/datasets/birds_kaggle_2020/official/train.csv
ln -s /share/cp/datasets/birds_kaggle_2020/external/xeno-canto/train_extended.csv
```
