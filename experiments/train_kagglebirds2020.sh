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
	#logsubdir="$logdir/${modelfile%.*}"  # tensorboard disabled for now
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
train 1 vanilla/defaults $data $model $metrics $training "$@"

# float16
data="--var dataset=kagglebirds2020"
model=
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/f16 $data $model $metrics $training "$@"

# ...
#data="--var dataset=kagglebirds2020"
#model="--var model.predictor.arch=conv2d:16@3x3,bn2d,lrelu,..."
#metrics="..."
#training="..."
#train 1 vanilla/defaults $data $model $metrics $training "$@"
