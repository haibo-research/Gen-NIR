# Generative Prior-Guided Neural Interface Reconstruction for 3D Electrical Impedance Tomography

![example](https://github.com/haibo-research/DiLO/blob/main/Utils/latent_space.png) 

## Abstract

We present a transformative **“solver-in-the-loop”** framework that bridges this divide by coupling a pre-trained 3D generative prior with a rigorous **Boundary Integral Equation (BIE)** solver. Our architecture enforces the governing elliptic PDE as a **hard constraint** at every optimization step, ensuring strict physical consistency. Simultaneously, we navigate a compact latent manifold of plausible geometries learned by a differentiable **Neural Diffeomorphic Flow (NDF)**, effectively regularizing the ill-posed problem through data-driven priors rather than heuristic smoothing. By propagating adjoint shape derivatives directly through the neural decoder, we achieve fast, stable convergence with dramatically reduced degrees of freedom, establishing a robust new paradigm for physics-constrained geometric discovery.

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/haibo-research/Gen-NIR.git
cd Gen-NIR
```
<br /> 

### 2) Set environment
Our environment relies on PyTorch and geometry processing libraries. Please ensure you have the appropriate CUDA Toolkit installed matching your PyTorch version:

```
conda create -n gen-nir python=3.9
conda activate gen-nir
pip install -r requirements.txt
```
<br /> 

### 3) Train 3D Generative Model
This project first utilizes a Neural Diffeomorphic Flow (NDF) to learn topology-preserving geometric distributions. Run the following command to train the generative prior:

```
python train/train_ndf.py
```
Note: This step trains a decoder capable of mapping the latent space to explicit 3D shapes.
<br /> 

### 4) Inference / 3D EIT Reconstruction
After obtaining the pre-trained generative prior, we couple it with the BIE solver to perform the 3D EIT reconstruction:

```
python optimize/EIT_NDF.py
```
This script initiates the solver-in-the-loop optimization process, adjusting the latent vector via backpropagated physical residuals to accurately reconstruct the 3D interface.
<br /> 

## Key Features
Physics as Hard Constraint: Integration of a BIE solver ensures every optimization step strictly adheres to the governing PDEs.

Generative Prior Regularization: Leverages NDF to handle complex topological changes and navigate a compact manifold of plausible geometries.

Differentiable Pipeline: End-to-end gradient backpropagation achieved through adjoint shape derivatives.
<br /> 

## Citation
If you find our work or this code useful for your research, please consider citing our paper:

```
@article{liu2026generative,
  title={Generative prior-guided neural interface reconstruction for 3D electrical impedance tomography},
  author={Liu, Haibo and Chen, Junqing and Lin, Guang},
  journal={Journal of Computational Physics},
  pages={114841},
  year={2026},
  publisher={Elsevier}
}
```
<br /> 

## Acknowledgements
This repository is built upon the 3D generative models codebase. We thank the authors for their open-source contributions.