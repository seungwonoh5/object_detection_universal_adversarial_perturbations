# Universal Adversarial Perturbations to Object Detection
Official implementation of "Resilience of Autonomous Vehicle Object Category Detections to Universal Adversarial Perturbations" (paper link).

This project evaluates the adversarial robustness of the state-of-the-art object detectors in the autonomous driving context. Our adversarial attack is based on the implementation of a research paper 'Universal Adversarial Perturbations to Object Detection', which it uses a variant of a projected gradient descent attack to create universal adversarial perturbations on object detection. This experiment is conducted on a variety of datasets, mainly MS COCO 2017 and 

## Getting Started
Clone this repository by running
```git clone https://github.com/seungwonoh5/Universal_Adversarial_Perturbation_Detection```

## Dependencies
This code requires the following:
* numpy==1.19.5
* pandas==1.1.5
* matplotlib==3.2.2
* tensorflow==2.4.1
* scikit-learn==0.22.2

run ```pip3 install -r requirements.txt``` to install all the dependencies.

To run the experiment, run ```python3 train.py.```

## What's Included
3 files with 2 scripts and 1 notebook file.
* data.py:this file provides all the data loading and preprocessing functions. (need to modify to use it for your own dataset)
* models.py: this file provides all of the decoder models in Keras. 
* utils.py: this file provides all the visualization and misc functions.

## Results
We perform extensive experiments on six datasets sequentially streaming and we show that an online setting continuously updating the model as every data block is processed leads to significant improvements over various state of the art models compared to the batch learning method that the model is fixed after training on the initial dataset and deploying for prediction. Specifically, on large-scale datasets that generally prove difficult cases for incremental learning, our approach delivers absolute gains as high as 19.1% and 7.4% on datasets, respectively.

## Contact
Author: Seungwon Oh - [aspiringtechsavvy@gmail.com](aspiringtechsavvy@gmail.com) or [soh1@terpmail.umd.edu](soh1@terpmail.umd.edu).a
To ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!
