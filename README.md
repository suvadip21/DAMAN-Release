## About
This repository provides the source-code for our implementation of *D*omain *A*dapted *M*ulti *A*moeba*N*et or __DAMAN__. Please follow the following links to access our publication in _IEEE Trans. on Medical Imaging_. [IEEExplore](https://ieeexplore.ieee.org/abstract/document/9870864), [preprint](https://www.researchgate.net/publication/363160645_Domain_Adapted_Multi-task_Learning_For_Segmenting_Amoeboid_Cells_in_Microscopy)

_This code is built on [this](https://github.com/JorisRoels/domain-adaptive-segmentation) repository. Consider citing the Y-Net paper as well if you are using this code._

-------------------------------

## Citing this work
If you are using this software, please cite the following articles:

[1] S. Mukherjee, R. Sarkar, M. Manich, E. Labruyère and J. -C. Olivo-Marin, "Domain Adapted Multi-task Learning For Segmenting Amoeboid Cells in Microscopy," in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3203022.

[2] S. Mukherjee, R. Sarkar, E. Labruyère and J. -C. Olivo-Marin, "A Min-Max Based Hyperparameter Estimation For Domain-Adapted Segmentation Of Amoeboid Cells," 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021, pp. 1869-1872, doi: 10.1109/ISBI48211.2021.9433864.

[3] R. Sarkar, S. Mukherjee, E. Labruyère and J. -C. Olivo-Marin, "Learning to segment clustered amoeboid cells from brightfield microscopy via multi-task learning with adaptive weight selection," 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 3845-3852, doi: 10.1109/ICPR48806.2021.9412641.

-------------------------------

## Usage
The entry-point to the codebase is the [main.py](main.py) file. The user has the option to
* Train the network on their own dataset
* Load a pre-trained model and use that for inference on their own data
* __Note__: _The provided pretrained model was trained on 256x256 images. Results on different resolutions could require fine-tuning
This model is trained (supervised) on brightfield, and domain adapted to fluorescence data.
The results are saved as 'inference.png'_
-------------------------------

## Data
If you wish to have access to the database of cell images used in this paper, please contact maria[dot]manich[at]pasteur[dot]fr.
A few images are provided to test our pre-trained model. 

-------------------------------

## Required packages
This implementation is based on Python 3.6.3 and uses the PyTorch library. The requirements are listed [here](requirements.txt). Alternatively, one could build a Docker container from [here](Dockerfile).


-------------------------------

## License
Copyright 2022 Bioimage Analysis Laboratory, Institut Pasteur

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
