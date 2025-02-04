# Implementation of Infant Vision

This project is developed as a submission for the Computational Visual Perception Course.

## Getting Started

1. Clone the repository. Create a virtual env using the command `python3 -m venv /path/to/new/virtual/environment`.
- Refer to the [documentation](https://docs.python.org/3/library/venv.html) for more details.
2. Install the package using `setup.py`:
   ```bash
   python setup.py install
   ```

## Task1: Visual Transforms Implemented

### 1.Visual Acuity
Visual acuity is the ability of the eye to discern fine details and distinguish objects clearly. 

**Infant Visual Acuity:** Poor visual acuity in infants, due to retinal and cortical immaturity, might adaptively support cortical development for broad spatial analysis.

<img src="Task1\output_images\va.png" width=750 height=180>

### 2. Contrast Sensitivity
Contrast sensitivity is the ability to detect differences in luminance or color between an object and its background, enabling the perception of edges, textures, and subtle variations in visual scenes.

**Infant Contrast Sensitivity:** 
* Newborns have low contrast sensitivity due to immature retinal and neural development, making it hard to detect subtle differences in light and dark.
* It improves rapidly in the first year as the visual system matures, with higher sensitivity to high-contrast patterns earlier than low-contrast details.

<img src="Task1\output_images\cs.png" width=750 height=180>

## Task2: Design of Curriculum Learning for Training
In this part of the project, we have applied image transformations, namely visual acuity and contrast sensitivity, to the TinyImageNet dataset and have trained them on a CustomResNet18 model using curriculum learning.

**Developmental Curriculum:** 

Curriculum learning, inspired by human education, gradually introduces tasks of increasing complexity or specificity, facilitating better learning progression. 
By gradually introducing age-specific data, the model systematically learns developmental visual patterns across age groups.

<img src="Task2\img\DevelopmentalCurriculum.png" width=900 height=400>

## Task3: --- Upcoming !! -------- 
