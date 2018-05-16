# NIMA: Neural Image Assessment

Implementation of the [NIMA model](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html) in TensorFlow.

<img src="https://github.com/master/nima/blob/master/nima.png?raw=true" height=100% width=100%>

Picture extracted from [1].

## Requirements

 * Python 3.5+
 * TensorFlow 1.6+

## Prerequisites

 * Download the Inception v2 weights from [TF-Slim models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
 * Download the AVA dataset from elsewhere
 * Convert AVA to TFRecords using `convert_ava.py` script:

```bash
./convert_ava.py --ava_dir <path to ava> --dataset_dir <path to dataset>
```

## Training

```bash
./train_eval_nima.py --dataset_dir <path to dataset> \
    --split_name=train \
    --log_dir <path to train dir> \
    --checkpoint_path <path to inception_v2.ckpt> \
    --checkpoint_exclude_scopes=InceptionV2/Logits
```

## Evaluation

```
./train_eval_nima.py --dataset_dir <path to dataset> \
    --split_name validation \
    --log_dir <path to train dir> \
    --eval \
    --max_epochs 1
```

## Results

The model has plateaued at 89% correlation after training for 20 epochs:

<img src="https://github.com/master/nima/blob/master/tensorboard.png?raw=true" height=100% width=100%>

## References

 1. Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment."
    IEEE Transactions on Image Processing (2018).

