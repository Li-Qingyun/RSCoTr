<p align="center">
  <h1 align="center">🐙 RSCoTr</h1>
</p>

[IEEE paper](https://ieeexplore.ieee.org/document/10401246)  |  [Model](https://huggingface.co/Qingyun/RSCoTr)

This repo hosts the offical implementation for the paper: **Co-training Transformer for Remote Sensing Image Classification, Segmentation and Detection**, *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, Qingyun Li, Yushi Chen, Xin He, and Lingbo Huang.

### Abstract

Several fundamental remote sensing (RS) image processing tasks, including classification, segmentation, and detection, have been set to serve for manifold applications. In the RS community, the individual tasks have been studied separately for many years. However, the specialized models were only capable of a single task. They lacked the adaptability for generalizing to the other tasks. Moreover, Transformer exhibits a powerful generalization capacity because it has the property of dynamic feature weighting. Hence, there is a large potential of a uniform Transformer to learn multiple tasks simultaneously, i.e., multi-task learning (MTL). An MTL Transformer can combine knowledge from different tasks by sharing a uniform network. In this study, a general-purpose Transformer, which simultaneously processes the three tasks, is investigated for RS MTL. To build a Transformer capable of the three tasks, an MTL framework named RSCoTr is proposed. The framework uses a shared encoder to extract multi-scale features efficiently and three task-specific decoders to obtain different results. Moreover, a flexible training procedure named co-training is proposed. The MTL model is trained with multiple general data sets annotated for individual tasks. The co-training is as easy as training a specialized model for a single task. It can be developed into different learning strategies to meet various requirements. The proposed RSCoTr is trained jointly with various strategies on three challenging data sets of the three tasks. And the results demonstrate that the proposed MTL method achieves state-of-the-art performance in comparison with other competitive approaches.

![RSCoTr](https://github.com/Li-Qingyun/RSCoTr/assets/79644233/f465f73b-4380-4879-a244-06fd33f80ce3)

## Enviroment Preparation

1. create a new env.

```shell
conda create -n rscotr python=3.8
conda activate rscotr
```

2. set cuda&gcc (recommanded, for current enviroment, you can also set it in ~/.bashrc)
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
touch $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
vim $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
```
write the following lines
```
# set cuda&gcc home
export CUDA_HOME=/mnt/share/cuda-11.3  # change this to <path to cuda-11.3>
export GCC_HOME=/mnt/share/gcc-7.3.0  # change this to <path to gcc-7.3>
# remove redundant cuda&gcc path
export PATH=$(echo "$PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
# set cuda&gcc path
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:$LD_LIBRARY_PATH
# set site-packages path
export SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
```
then `conda activate rscotr` to enable these env vars

3. install pytorch

```shell
# with conda
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
# with pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. install OpenMMLab

```shell
pip install openmim
mim install "mmcv-full==1.6.1"
mim install "mmdet==2.25.1"
mim install "mmsegmentation==0.28.0"
```

5. install other requirements

```shell
pip install -r requirement.txt
```

## Scripts Usages

The most important config is `configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py`, which means that training the MTL model using single level feature for classification (slvlcls), and swin-t-p4-w7_1x1 as backbone, and the three datasets.

```
# Train
python tools/train.py configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py --load-task-pretrain --work-dir work_dirs/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam
# Test
python tools/test.py configs/multi/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam.py work_dirs/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam/latest.pth --tasks cls det seg --work-dir work_dirs/MTL_slvlcls_swin-t-p4-w7_1x1_resisc&dior&potsdam
```

NOTE: The detection results are for validation set. An additional evaluation on test set is still required. You can use `tools/train_without_det_eval.py` which will pass the slow detection evaluation while training.  

`configs/multi/MTL_swin-t-p4-w7_1x1_resisc&dior&potsdam.py` is the version of using multi-level features for classification, as is reported in the paper, the performance did not get better.

`configs/multi/slvl_strategies` hosts the ablation experiments of iteration strategies.

`tools/inference_one_img.py` can inference the MTL model on one image for the three tasks.

## Contact and Acknowledge

Feel free to contact me through my email (21b905003@stu.hit.edu.cn) or [github issue](https://github.com/Li-Qingyun/RSCoTr/issues). I'll continue to maintain this repo.

The code is based on [OpenMMLab1.0 series toolkits](https://github.com/open-mmlab), including [mmcv](https://github.com/open-mmlab/mmcv), [mmcls](https://github.com/open-mmlab/mmcls), [mmdet](https://github.com/open-mmlab/mmdet) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Many modules refer to [MMF](https://mmf.sh/). The model architecture benefits from the insights of [DINO](https://arxiv.org/abs/2203.03605), [Mask2Former](https://arxiv.org/abs/2112.01527), [Deformable DETR](https://arxiv.org/abs/2010.04159), and [Swin Transformer](https://arxiv.org/abs/2103.14030). Thanks for their brilliant works.

Many thanks to the [Chinese WeChat article: 遥感与深度学习:《哈工大提出同时处理遥感分类/分割/目标检测的多任务学习框架RSCoTr, 基于Transformer》](https://mp.weixin.qq.com/s/9p_fXBM1vHKLGsXlCfJbdA). There are many high-quality Chinese articles about latest remote sensing papers in their channel.

## Citation

```
@article{li2024rscotr,
  title={Co-training transformer for remote sensing image classification, segmentation and detection},
  author={Li, Qingyun and Chen, Yushi and He, Xin and Huang, Lingbo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE},
  volume={62},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2024.3354783}
}
```
