# Image-based Reconstruction of Heterogeneous Media in the Presence of Multiple Light-Scattering
Ludwig Leonard and Rüdiger Westermann<br>Technical University of Munich<br>
| [Full paper] (https://) | 
![teaser](./docs/f_new_teaser-1.jpg)

<!--
## Abstract

Image-based reconstruction of a three-dimensional heterogeneous density field in the
presence of multiple scattering of light is intrinsically under-constrained. This leads to
reconstructions that look similar to the ground truth when rendered, but the recovered
field is often far off the real one. We shed light on the sources of uncertainty in the
reconstruction process which are responsible for this ambiguity, and propose the fol-
lowing approaches to improve the reconstruction quality: Firstly, we introduce a new
path sampling strategy, which yields more accurate estimates of the gradients of the
extinction field. Secondly, we build upon the observation that the variance in the loss
computation is one source of bias in the optimization process. To reduce this variance in
the primal estimator, we propose exploiting temporal coherence by reusing previously
rendered images. All this is coupled with a constraint on spatial object occupancy,
which restricts the problem to a reconstructed shape prior. In a number of examples we
demonstrate that compared to existing approaches the proposed reconstruction pipeline
leads to improved accuracy of the reconstructed density fields.
-->

This repository contains the official authors implementation associated with the paper "*Image-based Reconstruction of Heterogeneous Media in the Presence of Multiple Light-Scattering*".

## Citation

This project is under MIT License. If you found this project useful for your research please cite:

```bibtex
@article{leonard_and_westermann,
    author = {Ludwig Leonard and R\"udiger Westermann},
    title = {Image-based Reconstruction of Heterogeneous Media in the Presence of Multiple Light-Scattering},
    journal = {Computer & Graphics},
    year = {2024},
    publisher = {Elsevier},
    keywords = {differentiable rendering, inverse rendering, volumetric rendering}
}
```

## Rendervous

This repository is part of the [rendervous](https://github.com/rendervous) project and requires the rendervous library.

```shell
git clone https://github.com/rendervous/rendervous   
```

## Step-by-step

### Generating synthetic dataset

![dataset](./docs/dataset_example.png)

You can generate a synthetic dataset from a volume saved as a tensor.
In folder ```data``` there is an example of a volume compressed (```cloud_grid.zip```) and a HDR environment (```environment00.zip```).
Decompress at the root of ```data```. 
Execute the script ```generate_dataset.py```.

It takes around 01h15m in a GTX 3090 (preview images will be shown during the process).
The file ```cloud_dataset.pt``` should have be saved in the created folder ```datasets```.
The dataset contains the rendering for 80 different camera poses of 512x512 pixels, with 16K spp.

### DRT Pipeline

The baseline of our method is the sampling strategy implemented in
*Unbiased Inverse Volume Rendering With Differential Trackers* by Merlin, et al.

Executing the script ```reconstruct_ms_DRT.py``` we get the reconstruction using the aforementioned technique.
The constant BW_TECHNIQUE represents the sampling strategy used for the gradient propagation through the path.

- DRT: Differentiable Ratio-tracking
- SPS: Our Singular Path Sampler
- DRTDS: The defensive sampling used in DRT
- DRTQ: DRT in its quadratic form.
- DT: Vanila Delta Tracking without defensive strategy.

All these techniques can be tested in a pipeline that progressively reconstruct a medium from a coarse resolution to the final.
Using 10K iterations distributed among 4 levels.

The final reconstruction can be found at folder 
```python
f'./reconstructions/{DATASET}_drt_ms_{BW_TECHNIQUE}.pt'
```

### SPS Pipeline

The proposed pipeline in our paper requires an initial reconstruction with a Absorption-Emission model first.

The difference with the proposal in DRT is that our emission field relax in-scattered light in anisotropic case by using a Spherical Harmonics grid.

In order to generate such reconstruction has to be executed the script ```reconstruct_ae.py```. After this, a file at
```python
f'./reconstructions/{DATASET}_ae_{SH_LEVELS}.pt'
```
contains the reconstructed field of densities and emissions.
Our SPS pipeline will only make use of the densities.

Executing ```reconstruct_ms_SPS.py``` we reconstruct now using the three components proposed in our paper.

- An initial density from a relaxation of the in-scattering radiance.
- The singular path sampler.
- An exponential moving average to enhance the primal computation and avoid deviated gradients.

The final reconstruction can be found at folder 
```python
f'./reconstructions/{DATASET}_sps_ms_{BW_TECHNIQUE}.pt'
```

## Evaluation

The script ```eval_reconstructions.py``` allows to compare two reconstructions with respect to a reference volume and image.

![vol_comp](./docs/rec_vol_comparison.png)
![img_comp](./docs/rec_img_comparison.png)








