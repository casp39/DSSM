# Deep State-Space Model for Noise Tolerant Skeleton-Based Action Recognition

This repository provides **implementation code** for the paper:

> **Kazuki KAWAMURA, Takashi MATSUBARA, Kuniaki UEHARA**  
> **Deep State-Space Model for Noise Tolerant Skeleton-Based Action Recognition**,  
> *IEICE Transactions on Information and Systems*, **Vol.E103.D, No.6**, pp.1217–1225, 2020.  
> [Online ISSN 1745-1361](https://www.jstage.jst.go.jp/browse/transinf/-char/en), [Print ISSN 0916-8532](http://www.ieice.org/jpn/books/trans_inf.html).  
> [**PDF**](https://www.jstage.jst.go.jp/article/transinf/E103.D/6/E103.D_2019MVP0012/_pdf/-char/ja) | [**DOI**](https://doi.org/10.1587/transinf.2019MVP0012)

---

## Overview

This paper presents a **Deep State-Space Model (DSSM)** for robust skeleton-based action recognition.  
- **Skeleton data** (3D coordinates of human joints) has gained attention due to its relative insensitivity to appearance, viewpoint, and illumination.  
- However, **skeleton data often contains noise and missing values** when captured via depth sensors or pose estimation algorithms.  
- Our **DSSM** provides a **deep generative model** of time-series skeleton data that infers a **low-dimensional latent representation** (internal state), making the downstream classification more **robust** to noise and missing frames.

### Key Contributions
1. **Deep State-Space Model**  
   - Learns a **generative model** of skeletal motion.  
   - Incorporates a **skeleton transition model** (`p(x_t | z_t, x_{t-1})`) that leverages the previous pose (`x_{t-1}`) to reduce orientation/size biases.

2. **Robust to Noise and Missing Values**  
   - The generative approach better handles **incomplete** or **noisy skeleton data**.  
   - Outperforms a baseline RNN and some **state-of-the-art** methods when noise is present or frames are missing.

3. **Better Feature Extraction**  
   - The **latent states** serve as **informative features** for a final **sequence classifier** (e.g., BiLSTM).  
   - Demonstrates improved accuracy on **NTU RGB+D**, a large skeleton-action dataset.

---

### Abstract (Paper)
Action recognition using skeleton data (3D coordinates of human joints) is an attractive topic due to its robustness to the actor’s appearance, camera’s viewpoint, illumination, and other environmental conditions.
However, skeleton data must be measured by a depth sensor or extracted from video data using an estimation algorithm, and doing so risks extraction errors and noise.
In this work, for robust skeleton-based action recognition, we propose a deep state-space model (DSSM). The DSSM is a deep generative model of the underlying dynamics of an observable sequence.
We applied the proposed DSSM to skeleton data, and the results demonstrate that it improves the classification performance of a baseline method.
Moreover, we confirm that feature extraction with the proposed DSSM renders subsequent classifications robust to noise and missing values.
In such experimental settings, the proposed DSSM outperforms a state-of-the-art method.

### Citation
If you find this code or model useful in your research, please cite our paper:
```
@article{Kazuki KAWAMURA20202019MVP0012,
  title={Deep State-Space Model for Noise Tolerant Skeleton-Based Action Recognition},
  author={Kazuki KAWAMURA and Takashi MATSUBARA and Kuniaki UEHARA},
  journal={IEICE Transactions on Information and Systems},
  volume={E103.D},
  number={6},
  pages={1217-1225},
  year={2020},
  doi={10.1587/transinf.2019MVP0012}
}
```

