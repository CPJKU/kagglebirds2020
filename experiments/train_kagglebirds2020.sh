#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular base name for the weights and
# predictions files. Each single repetition checks if it was already run or is
# currently being run, creates a lockfile, trains the network, computes the
# predictions, and removes the lockfile. To distribute runs between multiple
# GPUs, just run this script multiple times with different --cuda-device=N.

here="${0%/*}"
outdir="$here/../results/kagglebirds2020"
logdir="$here/../logs/kagglebirds2020"

train_if_free() {
	modelfile="$1"
	echo "$modelfile"
	logsubdir="$logdir/${modelfile%.*}"
	modelfile="$outdir/$modelfile"
	mkdir -p "${modelfile%/*}"
	if [ ! -f "$modelfile" ] && [ ! -f "$modelfile.lock" ]; then
		for gpu in "$@" ''; do [[ "$gpu" == "--cuda-device="? ]] && break; done
		echo "$HOSTNAME: $gpu" > "$modelfile.lock"
		$PYTHON_COMMAND "$here"/../train.py "$modelfile" --logdir="$logsubdir" "${@:2}" #&& \
			#$PYTHON_COMMAND "$here"/../predict.py "$modelfile" "${modelfile%.*}.preds" --var batchsize=1 $gpu
		rm "$modelfile.lock"
	fi
}

train() {
	repeats="$1"
	name="$2"
	for (( r=1; r<=$repeats; r++ )); do
		train_if_free "$name"_r$r.mdl "${@:3}"
	done
}


# all defaults
data="--var dataset=kagglebirds2020"
model=
metrics=
training=
train 7 vanilla/defaults $data $model $metrics $training "$@"

# float16
data="--var dataset=kagglebirds2020"
model=
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/f16 $data $model $metrics $training "$@"

# trainable sharpness per class
data="--var dataset=kagglebirds2020"
model="--var model.global_pool=lmexxc:1"
metrics=
training=
train 1 vanilla/lmexxc $data $model $metrics $training "$@"
# same with float16
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/lmexxc_f16 $data $model $metrics $training "$@"

# PCEN
data="--var dataset=kagglebirds2020"
model="--var spect.magscale=pcen"
metrics=
training=
train 3 vanilla/pcen $data $model $metrics $training "$@"

# Weight loss by quality rating
data="--var dataset=kagglebirds2020"
model=
metrics="--var metrics._ce.weight_name=rating"
training=
train 1 vanilla/ratingweight $data $model $metrics $training "$@"

# log1p magnitude scaling
data="--var dataset=kagglebirds2020"
model="--var spect.magscale=log1px"
metrics=
training=
train 1 vanilla/log1px $data $model $metrics $training "$@"

# global average instead of log-mean-exp
data="--var dataset=kagglebirds2020"
model="--var model.global_pool=mean"
metrics=
training=
train 1 vanilla/meanpool $data $model $metrics $training "$@"

# 5-second snippets
data="--var dataset=kagglebirds2020 --var data.len_min=5 --var data.len_max=5"
model=
metrics=
training=
train 1 vanilla/len5 $data $model $metrics $training "$@"

# 5-second snippets, increased batch size
for batchsize in 32 64; do
  data="--var dataset=kagglebirds2020 --var data.len_min=5 --var data.len_max=5 --var batchsize=$batchsize"
  model=
  metrics=
  training=
  train 1 vanilla/len5_bs${batchsize} $data $model $metrics $training "$@"
done

# 10-, 15-, 20-second snippets with batch size 32
for len in 10 15 20; do
  data="--var dataset=kagglebirds2020 --var data.len_min=$len --var data.len_max=$len --var batchsize=32"
  model=
  metrics=
  training=
  train 1 vanilla/len${len}_bs32 $data $model $metrics $training "$@"
done


# shorter snippets with trained sharpness?


# ...
#data="--var dataset=kagglebirds2020"
#model="--var model.predictor.arch=conv2d:16@3x3,bn2d,lrelu,..."
#metrics="..."
#training="..."
#train 1 vanilla/defaults $data $model $metrics $training "$@"
