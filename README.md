# RSCoTr

This repo hosts the offical implementation for the paper: 

[Co-training Transformer for Remote Sensing Image Classification, Segmentation and Detection](https://github.com/Li-Qingyun/RSCoTr), *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*, Qingyun Li, Yushi Chen, Xin He, and Lingbo Huang.

The full text is also availiable at [ResearchGate](https://github.com/Li-Qingyun/RSCoTr).

# Abstract

Several fundamental remote sensing (RS) image processing tasks, including classification, segmentation, and detection, have been set to serve for manifold applications. In the RS community, the individual tasks have been studied separately for many years. However, the specialized models were only capable of a single task. They lacked the adaptability for generalizing to the other tasks. Moreover, Transformer exhibits a powerful generalization capacity because it has the property of dynamic feature weighting. Hence, there is a large potential of a uniform Transformer to learn multiple tasks simultaneously, i.e., multi-task learning (MTL). An MTL Transformer can combine knowledge from different tasks by sharing a uniform network. In this study, a general-purpose Transformer, which simultaneously processes the three tasks, is investigated for RS MTL. To build a Transformer capable of the three tasks, an MTL framework named RSCoTr is proposed. The framework uses a shared encoder to extract multi-scale features efficiently and three task-specific decoders to obtain different results. Moreover, a flexible training procedure named co-training is proposed. The MTL model is trained with multiple general data sets annotated for individual tasks. The co-training is as easy as training a specialized model for a single task. It can be developed into different learning strategies to meet various requirements. The proposed RSCoTr is trained jointly with various strategies on three challenging data sets of the three tasks. And the results demonstrate that the proposed MTL method achieves state-of-the-art performance in comparison with other competitive approaches.

![RSCoTr](https://github.com/Li-Qingyun/RSCoTr/assets/79644233/f465f73b-4380-4879-a244-06fd33f80ce3)
