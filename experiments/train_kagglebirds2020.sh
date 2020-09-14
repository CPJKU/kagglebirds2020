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

# median subtraction
data="--var dataset=kagglebirds2020"
model="--var spect.denoise=submedian"
metrics=
training=
train 1 vanilla/submedian $data $model $metrics $training "$@"

# different numbers of mel bands
for mb in 120 160 200; do
  data="--var dataset=kagglebirds2020"
  model="--var filterbank.num_bands=$mb"
  metrics=
  training=
  train 1 vanilla/mel${mb} $data $model $metrics $training "$@"
done

# different frames per second
for fps in 30 50 63 90 105; do
  data="--var dataset=kagglebirds2020"
  model="--var spect.fps=$fps"
  metrics=
  training=
  train 1 vanilla/fps${fps} $data $model $metrics $training "$@"
done

# PCEN with increased frontend learning rate
data="--var dataset=kagglebirds2020"
model="--var spect.magscale=pcen"
metrics=
training="--var train.first_params=frontend --var train.first_params.eta_scale=10"
train 2 vanilla/pcen_frontend-eta10 $data $model $metrics $training "$@"

# float16 with downmix augmentation
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
model=
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/rnddownmix_f16 $data $model $metrics $training "$@"

# float16 more median subtraction
data="--var dataset=kagglebirds2020"
model="--var spect.denoise=submedian"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 2 vanilla/submedian_f16 $data $model $metrics $training "$@"

# float16 with median subtraction and downmix augmentation
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
model="--var spect.denoise=submedian"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 2 vanilla/submedian_rnddownmix_f16 $data $model $metrics $training "$@"

# float16 resnet with median subtraction and downmix augmentation
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
model="--var spect.denoise=submedian --var model.predictor.arch=add[conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|crop2d:2,conv2d:64@1x1],add[bn2d,relu,conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|crop2d:2],pool2d:max@3x3,add[bn2d,relu,conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|crop2d:2,conv2d:128@1x1],add[bn2d,relu,conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|crop2d:2],bn2d,relu,conv2d:128@12x3,bn2d,lrelu,pool2d:max@5x3,conv2d:1024@1x9,bn2d,lrelu,dropout:0.5,conv2d:1024@1x1,bn2d,lrelu,dropout:0.5,conv2d:C@1x1"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 2 resnet1/submedian_rnddownmix_f16 $data $model $metrics $training "$@"

# float16 different context with median subtraction and downmix augmentation
for ctx in 3 5 7 11 13 15 17; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
  model="--var spect.denoise=submedian --var model.predictor.arch=conv2d:64@3x3,bn2d,lrelu,conv2d:64@3x3,bn2d,pool2d:max@3x3,lrelu,conv2d:128@3x3,bn2d,lrelu,conv2d:128@3x3,bn2d,lrelu,conv2d:128@17x3,bn2d,pool2d:max@5x3,lrelu,conv2d:1024@1x$ctx,bn2d,lrelu,dropout:0.5,conv2d:1024@1x1,bn2d,lrelu,dropout:0.5,conv2d:C@1x1"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_ctx${ctx}_f16 $data $model $metrics $training "$@"
done

# float16 resnet with median subtraction, downmix augmentation, pitch shifting
for pshift in 10 05 02; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
  model="--var spect.denoise=submedian --var filterbank.random_shift=0.$pshift"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_rndshift${pshift}_f16 $data $model $metrics $training "$@"
done

# float16 shakeshake with median subtraction and downmix augmentation
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
arch="add[shake[conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|conv2d:64@3x3,bn2d,relu,conv2d:64@3x3]|crop2d:2,conv2d:64@1x1],add[bn2d,relu,shake[conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|conv2d:64@3x3,bn2d,relu,conv2d:64@3x3]|crop2d:2],pool2d:max@3x3,add[bn2d,relu,shake[conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|conv2d:128@3x3,bn2d,relu,conv2d:128@3x3]|crop2d:2,conv2d:128@1x1],add[bn2d,relu,shake[conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|conv2d:128@3x3,bn2d,relu,conv2d:128@3x3]|crop2d:2],bn2d,relu,conv2d:128@12x3,bn2d,lrelu,pool2d:max@5x3,conv2d:1024@1x9,bn2d,lrelu,dropout:0.5,conv2d:1024@1x1,bn2d,lrelu,dropout:0.5,conv2d:C@1x1"
model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 2 shake1/submedian_rnddownmix_f16 $data $model $metrics $training "$@"

# float16 with median subtraction and downmix augmentation, different sharpness
for alpha in 10 5 0.1; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform"
  model="--var spect.denoise=submedian --var model.global_pool=lme:$alpha"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_lme${alpha/./}_f16 $data $model $metrics $training "$@"
done

# float16 with median subtraction and downmix augmentation on different splits
for seed in 10 20 30 40; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.split_seed=$seed"
  model="--var spect.denoise=submedian"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_f16_splitseed$seed $data $model $metrics $training "$@"
done

# float16 with median subtraction and downmix augmentation and background noise
for noise_prob in 0.25 0.5 0.75 1.0; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=$noise_prob"
  model="--var spect.denoise=submedian"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob${noise_prob/./}_f16 $data $model $metrics $training "$@"
done
for max_factor in 0.25 0.75 1.0; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=$max_factor"
  model="--var spect.denoise=submedian"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact${max_factor/./}_f16 $data $model $metrics $training "$@"
done

# float16 resnet with median subtraction, downmix augmentation, noise augmentation
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
model="--var spect.denoise=submedian --var model.predictor.arch=add[conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|crop2d:2,conv2d:64@1x1],add[bn2d,relu,conv2d:64@3x3,bn2d,relu,conv2d:64@3x3|crop2d:2],pool2d:max@3x3,add[bn2d,relu,conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|crop2d:2,conv2d:128@1x1],add[bn2d,relu,conv2d:128@3x3,bn2d,relu,conv2d:128@3x3|crop2d:2],bn2d,relu,conv2d:128@12x3,bn2d,lrelu,pool2d:max@5x3,conv2d:1024@1x9,bn2d,lrelu,dropout:0.5,conv2d:1024@1x1,bn2d,lrelu,dropout:0.5,conv2d:C@1x1"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 resnet1/submedian_rnddownmix_noiseprob10_noisemaxfact10_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, colored noise
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.mix_synthetic_noise.probability=1.0"
model="--var spect.denoise=submedian"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_synthnoiseprob10_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, groupnorm
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, balanced classes
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.class_sample_weights=equal"
model="--var spect.denoise=submedian"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_samplebalanced_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, 2d batchnorm frontend
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0"
model="--var spect.denoise=submedian --var spect.norm=batchnorm2d"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_frontendbn2d_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, quality filter
for min_rating in 3 3.5 4 4.5; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.min_rating_train=$min_rating"
  model="--var spect.denoise=submedian"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_minqual${min_rating/./}_f16 $data $model $metrics $training "$@"
done

# float16 with median subtraction, downmix augmentation, background noise, larger batches
for bs in 24 32;
do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var batchsize=$bs"
  model="--var spect.denoise=submedian"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_bs${bs}_f16 $data $model $metrics $training "$@"
done

# float16 resnet with median subtraction, downmix augmentation, background noise, groupnorm
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
model="--var spect.denoise=submedian --var model.predictor.arch=add[conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2,conv2d:64@1x1],add[groupnorm:16,relu,conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2],pool2d:max@3x3,add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2,conv2d:128@1x1],add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2],groupnorm:16,relu,conv2d:128@12x3,groupnorm:16,lrelu,pool2d:max@5x3,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 resnet1/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, log1p
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
model="--var spect.denoise=submedian --var model.predictor.arch=$arch --var spect.magscale=log1px"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_log1px_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, short inputs
for len in 5 10 15; do
  bs=$((16*30/len))
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.len_min=$len --var data.len_max=$len --var batchsize=$bs"
  arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
  model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_len${len}_bs${bs}_f16 $data $model $metrics $training "$@"
done

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, mixup
for alpha in 0.1 0.3; do
  for binarize in '' 'label_all'; do
    data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.mixup.apply_to=input,label_all --var data.mixup.alpha=$alpha --var data.mixup.binarize=$binarize"
    arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
    model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
    metrics=
    training="--var float16=1 --var float16.opt_level=O2"
    train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_mixup${alpha}${binarize/label_all/bin}_f16 $data $model $metrics $training "$@"
  done
done

# float16 resnet with median subtraction, downmix augmentation, background noise, groupnorm, log1p
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
arch="add[conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2,conv2d:64@1x1],add[groupnorm:16,relu,conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2],pool2d:max@3x3,add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2,conv2d:128@1x1],add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2],groupnorm:16,relu,conv2d:128@12x3,groupnorm:16,lrelu,pool2d:max@5x3,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
model="--var spect.denoise=submedian --var model.predictor.arch=$arch --var spect.magscale=log1px"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 resnet1/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_log1px_f16 $data $model $metrics $training "$@"

# pretrained PANN models with downmix augmentation, background noise
for blocks in 6 5 4 3; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
  model="--var model=pann --var model.num_blocks=$blocks"
  metrics=
  training=  # STFT does not support float16 with uncommon window lengths
  train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks$blocks $data $model $metrics $training "$@"
done

# pretrained PANN models with downmix augmentation, background noise, very careful tuning of pretrained part
for blocks in 6 5 4 3; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
  model="--var model=pann --var model.num_blocks=$blocks"
  metrics=
  training="--var train.first_params=frontend+predictor.pretrained --var train.first_params.eta_scale=0.01"
  train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks${blocks}_pannetascale001 $data $model $metrics $training "$@"
done

# pretrained PANN models with downmix augmentation, background noise, somewhat careful tuning of pretrained part
for blocks in 6; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
  model="--var model=pann --var model.num_blocks=$blocks"
  metrics=
  training="--var train.first_params=frontend+predictor.pretrained --var train.first_params.eta_scale=0.1"
  train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks${blocks}_pannetascale01 $data $model $metrics $training "$@"
done

# PANN model without pretrained weights
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
model="--var model=pann --var model.num_blocks=6 --var model.pretrained_weights="
metrics=
training=  # STFT does not support float16 with uncommon window lengths
train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_fromscratch $data $model $metrics $training "$@"

# float16 resnet with downmix augmentation, background noise, groupnorm
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
model="--var model.predictor.arch=add[conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2,conv2d:64@1x1],add[groupnorm:16,relu,conv2d:64@3x3,groupnorm:16,relu,conv2d:64@3x3|crop2d:2],pool2d:max@3x3,add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2,conv2d:128@1x1],add[groupnorm:16,relu,conv2d:128@3x3,groupnorm:16,relu,conv2d:128@3x3|crop2d:2],groupnorm:16,relu,conv2d:128@12x3,groupnorm:16,lrelu,pool2d:max@5x3,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
metrics=
training="--var float16=1 --var float16.opt_level=O2"
train 1 resnet1/rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_f16 $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, lower background label weight
for bgweight in 0.5 0.75; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.label_bg_weight=$bgweight"
  arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
  model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_bgweight${bgweight/./}_f16 $data $model $metrics $training "$@"
done

# pretrained PANN model with downmix augmentation, background noise, pitch shifting
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var filterbank.random_shift=0.10"
model="--var model=pann --var model.num_blocks=$blocks"
metrics=
training=  # STFT does not support float16 with uncommon window lengths
train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_rndshift10 $data $model $metrics $training "$@"

# pretrained PANN model with downmix augmentation, background noise, log1px
data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
model="--var model=pann --var model.num_blocks=$blocks --var magscale.trainable=1"
metrics=
training=  # STFT does not support float16 with uncommon window lengths
train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px $data $model $metrics $training "$@"

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, smaller freq range
for min_freq in 27.5 50 100; do
  for max_freq in 10000 8000 6000; do
    data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
    arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
    model="--var spect.denoise=submedian --var model.predictor.arch=$arch --var filterbank.min_freq=$min_freq --var filterbank.max_freq=$max_freq"
    metrics=
    training="--var float16=1 --var float16.opt_level=O2"
    train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_minf${min_freq/./}_maxf${max_freq}_f16 $data $model $metrics $training "$@"
  done
done

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, negative examples
for neg_prob in 0.01 0.025 0.05; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.mix_background_noise.noise_only_probability=$neg_prob"
  arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
  model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_negprob${neg_prob/./}_groupnorm16_f16 $data $model $metrics $training "$@"
done

# float16 with median subtraction, downmix augmentation, background noise, groupnorm, longer snippets
for len in 45 60; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.len_min=$len --var data.len_max=$len"
  arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
  model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_groupnorm16_len${len}_f16 $data $model $metrics $training "$@"
done

# pretrained PANN model with downmix augmentation, background noise, log1px, label smoothing
for alpha in 0.01 0.05; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0"
  model="--var model=pann --var model.num_blocks=$blocks --var magscale.trainable=1"
  metrics="--var _ce.label_smoothing=$alpha --var _ce.multilabel_dim=1"
  training=
  train 1 pann/rnddownmix_noiseprob10_noisemaxfact10_blocks6_log1px_lsmooth${alpha/./} $data $model $metrics $training "$@"
done

# float16 with median subtraction, downmix augmentation, amplitude-controlled background noise, groupnorm
for max_amp in 1.0 0.75 0.5; do
  data="--var dataset=kagglebirds2020 --var data.downmix=random_uniform --var data.mix_background_noise.probability=1.0 --var data.mix_background_noise.max_factor=1.0 --var data.mix_background_noise.max_amp=$max_amp"
  arch="conv2d:64@3x3,groupnorm:16,lrelu,conv2d:64@3x3,groupnorm:16,pool2d:max@3x3,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@3x3,groupnorm:16,lrelu,conv2d:128@17x3,groupnorm:16,pool2d:max@5x3,lrelu,conv2d:1024@1x9,groupnorm:16,lrelu,dropout:0.5,conv2d:1024@1x1,groupnorm:16,lrelu,dropout:0.5,conv2d:C@1x1"
  model="--var spect.denoise=submedian --var model.predictor.arch=$arch"
  metrics=
  training="--var float16=1 --var float16.opt_level=O2"
  train 1 vanilla/submedian_rnddownmix_noiseprob10_noisemaxfact10_noisemaxamp${max_amp/./}_groupnorm16_f16 $data $model $metrics $training "$@"
done
