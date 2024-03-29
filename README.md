# Resilience of Autonomous Vehicle Object Category Detection to Universal Adversarial Perturbations
Official Pytorch implementation of our [paper](https://arxiv.org/pdf/2107.04749.pdf), “Resilience of Autonomous Vehicle Object Category Detection to Universal Adversarial Perturbations”, accepted to appear in the IEEE International IOT, Electronics, and Mechatronics (IEMTRONICS ’21) Conference.

This project evaluates the adversarial robustness of one of the state-of-the-art object detectors, Faster-RCNN, in the autonomous driving context. Our adversarial attack is based on the implementation of a research paper 'Universal Adversarial Perturbations to Object Detection', which it uses a variant of a projected gradient descent attack to create universal adversarial perturbations against object detection. This experiment is conducted on a variety of datasets, mainly a subset of COCO2017 training with 5 autonomous driving-related categories: car, truck, people, stop sign, and traffic light.

## Installation
Clone this repository by running ```git clone https://github.com/seungwonoh5/Universal_Adversarial_Perturbation_Detection```

## Dependencies
This code requires the following:
* numpy==1.19.5
* pandas==1.1.5
* matplotlib==3.2.2
* tensorflow==2.4.1
* scikit-learn==0.22.2

run ```pip3 install -r requirements.txt``` to install all the dependencies.

## What's Included
Inside the repo, there are mainly 2 scripts and 2 notebook files:
* visualization.py/ipynb: this notebook visualizes the experimental results.
* uap_autonomous.py/ipynb: this script provides running the experiment with specific adversarial attacks, object detection model, and dataset.

## Getting Started
execute ```run.py ``` to choose an adversarial attack to attack an object detection model in the Detectron2 Library.

## Results
We perform extensive experiments on six datasets sequentially streaming and we show that an online setting continuously updating the model as every data block is processed leads to significant improvements over various state of the art models compared to the batch learning method that the model is fixed after training on the initial dataset and deploying for prediction. Specifically, on large-scale datasets that generally prove difficult cases for incremental learning, our approach delivers absolute gains as high as 19.1% and 7.4% on datasets, respectively.



## Contact 
[nayeem@umd.edu](nayeem@umd.edu) or [wonoh90@gmail.com](wonoh90@gmail.com) to ask questions or report issues, please open an issue on the issues tracker. Discussions, suggestions and questions are welcome!

If you find this code useful in your research, please consider citing:
"""
    Mohammad Nayeem Teli, and Seungwon Oh. “Resilience of Autonomous Vehicle Object Category Detection
    to Universal Perturbations”, accepted to appear in the IEEE International IOT, Electronics, and Mechatronics (IEMTRONICS ’21) Conference
"""
