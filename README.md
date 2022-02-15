# PUNET
Hi Thank you for viewing this Project.
![alt text](https://github.com/OmarBoudraa/PUNET/blob/main/images/punet.png)

I shall write a few steps down for you to run this project in your machine.
Please note that I shall consider that you have git, pip3, and virtualenv.

Steps:
* Setup a new virtualenv using: `virtualenv -p python3 punet`
* Install some essential packages using:
	- `pip3 install numpy`
	- `pip3 install pandas`
	- `pip3 install opencv-python`
	- `pip3 install tensorflow-gpu` (or if you do not have a GPU then, `pip3 install tensorflow`)
	- `pip3 install keras`
* Now, Clone this repository using `git clone https://github.com/OmarBoudraa/PUNET`
* Go to the directory of project: `cd PUNET`
* Now, untar your dataset (IAM, ESP or others) in `images` & `xml` folders |
	In this example, we make use of famous GW dataset.
* We are now ready to execute the model. Execute: `python punet.py`

Please note, if you do not have a GPU in your computer, you should comment the following lines:
- punet_classifier.py => lines: {12-16}, 18, 107

If you have a GPU but do not have multiple GPUs in your system, please comment like:
- punet_classifier.py => line: 107

If you use the code for scientific purposes, please cite
```
@article{BoudraaOPRL2022,
   author = {O. Boudraa and D. Michelucci and W.K. Hidouci},
   title = {{PUNet: Novel and efficient deep neural network architecture for handwritten documents word spotting}},
   journal = {Pattern Recognition Letters},
   volume = {155},
   pages = {19-26},
   year = {2022}
}
