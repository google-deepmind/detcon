# Code for DetCon

This repository contains code for the ICCV 2021 paper
["Efficient Visual Pretraining with Contrastive Detection"](https://arxiv.org/abs/2103.10957)
by Olivier J. Hénaff, Skanda Koppula, Jean-Baptiste Alayrac, Aaron van den Oord,
Oriol Vinyals, João Carreira.

This repository includes sample code to run pretraining with DetCon. In
particular, we're providing a sample script for generating the Felzenzwalb
segmentations for ImageNet images (using `skimage`) and a pre-training
experiment setup (dataloader, augmentation pipeline, optimization config, and
loss definition) that describes the DetCon-B(YOL) model described in the paper.
The original code uses a large grid of TPUs and internal infrastructure for
training, but we've extracted the key DetCon loss+experiment in this folder so
that external groups can have a reference should they want to explore
a similar approaches.

This repository builds heavily from the
[BYOL open source release](https://github.com/deepmind/deepmind-research/tree/master/byol),
so speed-up tricks and features in that setup may likely translate to the code
here.

## Running this code

Running `./setup.sh` will create and activate a virtualenv and install all
necessary dependencies. To enter the environment after running `setup.sh`, run
`source /tmp/detcon_venv/bin/activate`.

Running `bash test.sh` will run a single training step on a mock
image/Felzenszwalb mask as a simple validation that all dependencies are set up
correctly and the DetCon pre-training can run smoothly. On our 16-core machine,
running on CPU, we find this takes around 3-4 minutes.

A TFRecord dataset containing each ImageNet image, label, and its corresponding
Felzenszwalb segmentation/mask can then be generated using the
`generate_fh_masks` Python script. You will first have to download two pieces of
ImageNet metadata into the same directory as the script:

`wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_metadata.txt`
`wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt`

And to run the multi-threaded mask generation script:

```
python generate_fh_masks_for_imagenet.py -- \
--train_directory=imagenet-train \
--output_directory=imagenet-train-fh
```

This single-machine, multi-threaded version of the mask generation script takes
2-3 days on a 16-core CPU machine to complete CPU-based processing of the
ImageNet training and validation set. The script assumes the same ImageNet
directory structure as
[github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py](https://github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py)
(more details in the link).

You can then run the main training loop and execute multiple DetCon-B training
steps by running from the parent directory the command:

```bash
python -m detcon.main_loop \
  --dataset_directory='/tmp/imagenet-fh-train' \
  --pretrain_epochs=100`
```

Note that you will need to update the `dataset_directory` flag, to point to the
generated Felzenzwalb/image TFRecord dataset previously generated. Additionally,
to use accelerators, users will need to install the correct version of jaxlib
with CUDA support.

## Pre-trained checkpoints

For convenience, we're providing an ImageNet-pretrained [ResNet-50](https://storage.googleapis.com/dm-detcon/resnet50_detcon_b_imagenet_1k.npy) and [ResNet-200](https://storage.googleapis.com/dm-detcon/resnet200_detcon_b_imagenet_1k.npy) pre-trained using DetCon. We also provide a number of ResNet-50 COCO-pretrained checkpoints available in the same [GCS bucket]((https://storage.googleapis.com/dm-detcon/). A Colab demonstrating how to load the model weights and run a forward pass on the loaded model on a sample image is linked [here](https://colab.research.google.com/drive/1Gd3sxOJXENo74iPz5TlywEcsfXX1gB8W?usp=sharing).

## Citing this work

If you use this code in your work, please consider referencing our work:

```
@article{henaff2021efficient,
  title={{Efficient Visual Pretraining with Contrastive Detection}},
  author={H{\'e}naff, Olivier J and Koppula, Skanda and Alayrac, Jean-Baptiste and Oord, Aaron van den and Vinyals, Oriol and Carreira, Jo{\~a}o},
  journal={International Conference on Computer Vision},
  year={2021}
}
```

## Disclaimer

This is not an officially supported Google product.
