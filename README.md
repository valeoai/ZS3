# zero_shot_semantic_segmentation

## Paper
![](./teaser.png)

[Zero-Shot Semantic Segmentation](https://arxiv.org/pdf/1906.00817.pdf)  
 [Maxime Bucher](https://maximebucher.github.io/), [Tuan-Hung Vu](https://tuanhungvu.github.io/) , [Matthieu Cord](http://webia.lip6.fr/~cord/), [Patrick Pérez](https://ptrckprz.github.io/)  
 valeo.ai, France  
 Neural Information Processing Systems (NeurIPS) 2019

If you find this code useful for your research, please cite our [paper](https://arxiv.org/pdf/1906.00817.pdf):

```
@inproceedings{bucher2019zero,
  title={Zero-Shot Semantic Segmentation},
  author={Bucher, Maxime and Vu, Tuan-Hung and Cord, Mathieu and P{\'e}rez, Patrick},
  booktitle={NeurIPS},
  year={2019}
}
```

## Abstract
Semantic segmentation models are limited in their ability to scale to large numbers of object classes. In this paper, we introduce the new task of zero-shot semantic segmentation: learning pixel-wise classifiers for never-seen object categories with zero training examples. To this end, we present a novel architecture, ZS3Net, combining a deep visual  segmentation model with an approach to generate visual representations from semantic word embeddings. By this way, ZS3Net addresses pixel classification tasks where both seen and unseen categories are faced at test time (so called "generalized" zero-shot classification). Performance is further improved by a self-training step that relies on automatic pseudo-labeling of pixels from unseen classes. On the two standard segmentation datasets, Pascal-VOC and Pascal-Context, we propose zero-shot benchmarks and set competitive baselines. For complex scenes as ones in the Pascal-Context dataset, we extend our approach by using a graph-context encoding to fully leverage spatial context priors coming from class-wise segmentation maps.

## Code
Coming soon!
