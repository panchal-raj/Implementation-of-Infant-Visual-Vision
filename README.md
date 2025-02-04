# Implementation of Infant Vision

This project is developed as a submission for the Computational Visual Perception Course.

## Getting Started

1. Clone the repository. Create a virtual env using the command `python3 -m venv /path/to/new/virtual/environment`.
2. Install the package using `setup.py`:
  
   ```bash
   python setup.py install
   ```

## Task1: Visual Transforms Implemented

### 1. Visual Acuity
Visual acuity is the ability of the eye to discern fine details and distinguish objects clearly. 

**Infant Visual Acuity:** Poor visual acuity in infants, due to retinal and cortical immaturity, might adaptively support cortical development for broad spatial analysis.

<img src="Task1\output_images\va.png" width=750 height=180>

### 2. Contrast Sensitivity
Contrast sensitivity is the ability to detect differences in luminance or color between an object and its background, enabling the perception of edges, textures, and subtle variations in visual scenes.

**Infant Contrast Sensitivity:** 
* Newborns have low contrast sensitivity due to immature retinal and neural development, making it hard to detect subtle differences in light and dark.
* It improves rapidly in the first year as the visual system matures, with higher sensitivity to high-contrast patterns earlier than low-contrast details.

<img src="Task1\output_images\cs.png" width=750 height=180>

Details: Reports/Task1

## Task2: Design of Curriculum Learning for Training
In this part of the project, we have applied image transformations, namely visual acuity and contrast sensitivity, to the TinyImageNet dataset and have trained them on a CustomResNet18 model using curriculum learning.

**Developmental Curriculum:** 

Curriculum learning, inspired by human education, gradually introduces tasks of increasing complexity or specificity, facilitating better learning progression. 
By gradually introducing age-specific data, the model systematically learns developmental visual patterns across age groups.

<img src="Task2\img\DevelopmentalCurriculum.png" width=932 height=469>

Details: Reports/Task2

## Task3: Evaluation of Neural Networks with RDMs and NED
In Part 3, we evaluate trained artificial neural networks using Representational Dissimilarity Matrices (RDMs) and compare them to different regions of interest (ROIs) in the brain responsible for vision, utilizing the Neural Encoding Dataset (NED).

**Analysis of Feature Representations of layers Using RDMs:** 
Representational Dissimilarity Matrices (RDMs) measure differences in feature representations across models or network layers by computing pairwise correlation distances.

<img src="output\RDMs_Images\both_transforms_conv1.png" width=300 height=300>    <img src="output\RDMs_Images\both_transforms_conv1.png" width=300 height=300>    <img src="output\RDMs_Images\both_transforms_conv1.png" width=300 height=300>

**Comparision of Neural Networks with Brain Data:**
We evaluated our networks using the Neural Encoding Dataset (NED), which includes pre-generated fMRI responses from the Natural Scenes Dataset (NSD) across multiple brain regions of interest (ROIs) in different subjects.

Details: Reports/Task3
