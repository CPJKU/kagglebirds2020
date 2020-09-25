7th Place Solution for the Cornell Birdcall Identification Kaggle Challenge
===========================================================================

Implementation of the submission of team "[CP-JKU] Three Geese and a GAN" of
the [Institute of Computational
Perception](https://https://www.jku.at/en/institute-of-computational-perception/)
at Johannes-Kepler-University Linz, Austria, to the [Cornell Birdcall
Identification Challenge](https://www.kaggle.com/c/birdsong-recognition/) run
at Kaggle from July to September 2020.

This submission reached place 7 of 1390 on the [private
leaderboard](https://www.kaggle.com/c/birdsong-recognition/leaderboard), and
can be modified to score between places 1 and 2.

For a detailed explanation of the methods employed here, please see the [7th
place solution
writeup](https://www.kaggle.com/c/birdsong-recognition/discussion/183571) as
posted on Kaggle.


Prerequisites
-------------

### Software

You will need Python 3 with at least the following modules:
* numpy
* scikit-learn
* torch>=1.5.1
* tqdm
* pandas

For some features, additional modules are required:
* tensorboard
* apex
* matplotlib

With pip, prerequisites can be installed with:
```bash
pip3 install numpy scikit-learn tqdm pandas matplotlib
```

Similar with conda:
```bash
conda install numpy scikit-learn tqdm pandas matplotlib
```

Pytorch should be installed following the instructions on
[pytorch.org](https://pytorch.org). You will need Pytorch 1.5.1 or later.

Tensorboard can be installed with `pip3 install tensorboard` or
`conda install -c conda-forge tensorboard`.

Apex currently needs to be installed via pip (even when using conda):
```
TORCH_CUDA_ARCH_LIST="6.1;7.5" pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/master.zip
```
`TORCH_CUDA_ARCH_LIST` lists the compute capabilities to compile for; it only
needs to be set if the installation is shared across multiple machines with
different GPUs (by default, it compiles for all GPUs on the current machine).

Apart from Python, you will need `ffmpeg`. On Debian and Ubuntu, this can be
installed with `sudo apt install ffmpeg libavcodec-extra`.

### Dataset

To prepare the dataset for training, follow the instructions in
[`definitions/datasets/kagglebirds2020/README.md`](./definitions/datasets/kagglebirds2020/README.md).

### PANN weights

To prepare the pretrained PANN model for training, follow the instructions in
[`definitions/models/pann/README.md`](./definitions/models/pann/README.md).


Training
--------

To train the models needed for one of the submissions, run:
```bash
OMP_NUM_THREADS=1 experiments/train_kagglebirds2020.sh --cuda-device=0
```
Append `--var data.num_workers=4 --var data.pin_memory=1` to parallelize data
loading over multiple subprocesses if your GPU is not fully utilized.
If you do not want to log experiments with TensorBoard, append `--logdir=''`.
To run experiments in parallel on multiple GPUs, simply run the script multiple
times with different CUDA device indices (the ones already running or finished
will be skipped).

The training script contains all experiments ran up to the submission deadline,
but has most of them commented out.


Prediction
----------

It is possible to compute predictions locally for a test set following the
challenge's test set layout (for example, the
[birdcall-check](https://www.kaggle.com/shonenkov/birdcall-check) or
[birdcall-check-2](https://www.kaggle.com/janschl/birdcall-check-2) dataset).
To do so, run the `predict_kagglebirds2020.py` script, pointing it to the
location of the dataset and all the model files you want to ensemble:
```bash
OMP_NUM_THREADS=1 ./predict_kagglebirds2020.py --threshold=0.5 \
  --threshold-convert=logit --var data.len_max=20 --local-then-global \
  --local-then-global-overlap=0.5 --filter-local-by-global \
  --filtered-threshold=0.1 --var model.pretrained_weights= \
  --train-csv=definitions/datasets/kagglebirds2020/train.csv \
  --test-csv=/share/cp/datasets/birds_kaggle_2020/external/birdcall_check2/test.csv \
  --test-audio=/share/cp/datasets/birds_kaggle_2020/external/birdcall_check2/test_audio \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px_negprob001_r1.mdl \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_noisemaxamp10_blocks6_log1px_negprob001_r1.mdl \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px_r1.mdl \
  /tmp/predictions.csv
```
This includes several options for the inference. The most important one seems
to be `--filtered-threshold=...`: Lowering it will allow more local detections
to come through for the species that were found to be present in the recording
with the larger global `--threshold=...` and longer pooling window of length
`--var data.len_max=...` (in seconds). The challenge submissions were done with
`--filtered-threshold=0.3`, but a value of 0.2, 0.1 or 0.05 gives a
progressively better balance of precision and recall, and a higher f1-score.


Evaluation
----------

It is also possible to evaluate predictions produced as above against a ground
truth "perfect submission" as included in the
[birdcall-check-2](https://www.kaggle.com/janschl/birdcall-check-2) dataset:
```bash
./eval_kagglebirds2020.py \
  --train-csv=definitions/datasets/kagglebirds2020/train.csv \
  /share/cp/datasets/birds_kaggle_2020/external/birdcall_check2/perfect_submission.csv \
  /tmp/predictions.csv
```


Submission
----------

The submission to Kaggle is based on the same prediction script used above. The
idea is to upload the code and model weights as a private dataset, then have a
simple Kaggle kernel extract them and run the prediction script. Two shell
scripts are included to help with this. The first one just collects the files:
```bash
./kaggle_pack.sh \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px_negprob001_r1.mdl \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_noisemaxamp10_blocks6_log1px_negprob001_r1.mdl \
  results/kagglebirds2020/pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px_r1.mdl
```
The code is stored in `kaggle_upload/kaggle_py.tgz` and the model weights and
configurations in `kaggle_upload/kaggle_mdl.tgz`.

You can either directly upload them as a private dataset, or automate this even
further: Install the Kaggle API client (`pip3 install kaggle`), create an API
token (see https://www.kaggle.com/docs/api), create a private dataset on
Kaggle, then update the `kaggle_upload/dataset-metadata.json` file. Once done,
you can call `kaggle_upload.sh "some comment"` to create a new version of the
dataset using what's in the `kaggle_upload/` directory.

Finally, create a new Notebook on Kaggle, add both your own dataset and the
[birdcall-check](https://www.kaggle.com/shonenkov/birdcall-check) dataset,
import the notebook from `kaggle_submit.ipynb` (or copy/paste the commands),
and do "Save version and run all". Afterwards it should be ready for
submission. Whenever you have a new ensemble, just rerun `./kaggle_pack.sh` and
`./kaggle_upload.sh`, open the notebook, check for a new version for your
private dataset, then do "Save version and run all" again.
