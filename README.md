# Generating Informative Responses with Controlled Sentence Function

## Introduction

Sentence function is a significant factor to achieve the purpose of the speaker. In this paper, we present a novel model to generate informative responses with controlled sentence function. Given a user post and a sentence function label, our model is to generate a response that is not only coherent with the specified function category, but also informative in content.

This project is a tensorflow implementation of our work.

## Dependencies
	
* Python 2.7
* Numpy
* Tensorflow 1.3.0

## Quick Start

* Dataset

	Our dataset contains single-turn post-response pairs with corresponding sentence function labels. The sentence function labels of responses have been automatically annotated by a self-attentive classifier.

	Please download the [Chinese Dialogue Dataset with Sentence Function Labels](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense) to data directory.

* Train

	```python main.py	```

* Test

	```python main.py --is_train=False --inference_path='xxx' --inference_version='yyy'	```

	You can test the model using this command. You may set the directory of test set with inference_path and the checkpoint to be used with inference_version. The generation result will be output to the 'xxx.out' file.


## Details

### Training

You can change the model parameters using:

	--symbols xxx           size of full vocabulary
	--topic_symbols xxx			size of topic vocabulary
	--full_kl_step xxx			parameter of kl annealing
	--units xxx 				    size of hidden units
	--embed_units xxx			  dimension of word embedding
	--batch_size xxx 			  batch size in training process
	--per_checkpoint xxx 		steps to save and evaluate the model
	--data_dir xxx				  data directory
	--train_dir xxx				  training directory


## Paper

Pei Ke, Jian Guan, Minlie Huang, Xiaoyan Zhu.
[Generating Informative Responses with Controlled Sentence Function.](http://aclweb.org/anthology/P18-1139)  
ACL 2018, Melbourne, Australia.

**Please kindly cite our paper if this paper and the code are helpful.**


## License

Apache License 2.0
