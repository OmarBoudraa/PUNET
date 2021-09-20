Hi Thank you for viewing this repository.

I shall write a few steps down for you to run this project in your machine.
Please note that I shall consider that you have git, pip3, and virtualenv.

Steps:
* Setup a new virtualenv using: `virtualenv -p python3 punet_keras`
* Install some essential packages using:
	- `pip3 install numpy`
	- `pip3 install pandas`
	- `pip3 install opencv-python`
	- `pip3 install tensorflow-gpu` (or if you do not have a GPU then, `pip3 install tensorflow`)
	- `pip3 install keras`
* Now, Clone this repository using `git clone https://github.com/OmarBoudraa/punet_keras`
* Go to the directory of project: `cd punet_keras`
* Now, untar the dataset present in `word` & `xml` folders using:
	- `tar -xvf words/words.tgz`
	- `tar -xvf xml/xml.tgz`
* We are now ready to execute the model. Execute: `python punet.py`

Please note, if you do not have a GPU in your computer, you should comment the following lines:
- punet_classifier.py => lines: {13-17}, 19, 75

If you have a GPU but do not have multiple GPUs in your system, please comment like:
- punet_classifier.py => line: 75

If you use the code for scientific purposes, please cite
```
@article{BoudraaOPRL2021,
   booktitle = {Pattern Recognition Letters},
   author = {Boudraa O., Michelucci D., and Hidouci W.K.},
   title = {{PUNet: Novel and efficient deep neural network architecture for handwritten documents word spotting}},
   year = {2021}
}
