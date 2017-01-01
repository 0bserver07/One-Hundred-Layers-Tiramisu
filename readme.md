### The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation:
---

*Work In Progress, Theano Recurrsion Error*


What is The One Hundred Layers Tiramisu?

* A state of art (as in Jan 2017) Semantic Pixel-wise Image Segmentation model that consists of a fully deep convolutional blocks with downsampling, skip-layer then to Upsampling architecture. 
* An extension of DenseNets to deal with the problem of semantic segmentation.

 **Fully Convolutional DensNet** = **(Dense Blocks + Transition Down Blocks)** + **(Bottleneck Blocks)** + **(Dense Blocks + Transition Up Blocks)** +  **Pixel-Wise Classification** layer

 ![model](./imgs/tiramisu-103.png)



##### *[The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation (Simon Jégou, Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua Bengio) arXiv:1611.09326 cs.CV](https://arxiv.org/abs/1611.09326)*
 	

#### Model Strucure:
-----

* DenseBlock: 
	`BatchNormalization` + `Activation [ Relu ]` + `Convolution2D` + `Dropout` 

* TransitionDown: 
	`BatchNormalization` + `Activation [ Relu ]` + `Convolution2D` + `Dropout` + `MaxPooling2D`

* TransitionUp: 
	`Deconvolution2D` (Convolutions Transposed)

 ![model-blocks](./imgs/tiramisu-blocks.png)

-----
### Repo (explanation):
---

* Download the CamVid Dataset as explained below:
	* Use the `data_loader.py` to crop images to `224, 224` as in the paper implementation.
* run `python model-tirmasu.py` for now to generate the FC-Dense103 Layers Model
* run `python train-tirmasu.py` to start training:
	* `Theano` can be changed inside.
	* Saves best checkpoints for the model and `data_loader` included for the `CamVidDataset`
* `helper.py` contains two methods `normalized` and `one_hot_it`, currently for the CamVid Task

### Dataset:
---

1. In a different directory run this to download the [dataset from original Implementation](https://github.com/alexgkendall/SegNet-Tutorial).
	* `git clone git@github.com:alexgkendall/SegNet-Tutorial.git`
	* copy the `/CamVid` to here, or change the `DataPath` in `data_loader.py` to the above directory
2. The run `python data_loader.py` to generate these two files:
	
	* `/data/train_data.npz/` and `/data/train_label.npz`
	* This will make it easy to process the model over and over, rather than waiting the data to be loaded into memory.



----


### To Do:
----

	[ ] FC-DenseNet 103
	[ ] FC-DenseNet 53
	[ ] Replicate Test Accuracy CamVid Task
	[ ] Replicate Test Accuracy GaTech Dataset Task
	[ ] Requirements


* Original Results Table:

	 ![model-results](./imgs/original-result-table.png)


	