# Latent Space Learning for GPI

This repository explores several different approaches to the the problem of giving time consistency to latent space embeddings, in order to enforce smoothness conditions on the embedded trajectories defined by videos. This framework aims to extend the work of [1](#bibliography), in which robotic tasks are embedded as trajectories with the aim to introduce a new framework for imitation learning, and aims to extend it to general vision-driven tasks with ML models such as AutoEncoders and Pretrained Vision and Multimodal Models.   

The [Jupyter Notebook File](Notebook.ipynb) contains examples of the methods implemented and relative results when applied to the Push-T task simulation, for which the data was shared in [2](#bibliography), in which a robotic arm aims to move and position a grey T-shaped block on a give position and orientation (given by a green t-shape area on the working environment).

## AutoEncoder Based Models
The first class of models is given by AutoEncoder based models. For this class of models, other than applying the standard approach, I implemented two additional classes of models which I defined Time Consistent AutoEncoder (which adds the latent space distance between two consecutive embeddings to its optimization loss) and a BiLevel Time AutoEncoder (which instead defines a secondary Neural Network as a "fictional" time advance operator, and aims to minimize the difference between advancing in time and then embedding and viceversa).   

The following plot summarizes the results on the PushT task, showing the strength of this approach compared to the standard one in terms of latent-space structure.  

![alt text](pictures/autoencoders_results.png)

The implementations for these methods can be found in the [autoencoders](src/autoencoder.py) and [autoencoder training utils](src/autoencoder_training_utils.py) files.


## Pretrained Vision Model Based Approaches
The second approach explored consists of using Pretrained Vision models to generate latent space embeddings, and possibly apply fine-tuning approaches in order to achieve an higher level of structure.    
In particular, my approach aims to perform fine-tuning with a small fine-tuning head trained in an unsupervised fashion, rather than tuning all parameters in the network with some labeled data.   

My approach is based on contrastive learning and uses exponentially decaying weights in order to define positive pairs within the entire demonstrations, performing a temporal alignement of the embeddings in latent space.   

Once again, the results are summarized in the following plot:

![alt text](pictures/pretrained_results.png)

The implementations for these methods can be found in the [embedding head](src/embedding_head.py) and [fine tuning utils](src/fine_tuning_utils.py) files.

## Multimodal Model Based Approaches
The final approach explored uses Text-Vision multimodal models to extract prompt-guided embeddings using - once again - pretrained models. In this case, the supervision is performed at an high level using textual prompts to guide the vision model through cross-attention to extract more meaningful information from the images. 

The following plot compares the results of an empty prompt with generic and task specific prompts:

![alt text](pictures/multimodal_results.png)



## Bibliography

- 1. Yiming Li & Sylvain Calinon (2025). [From Movement Primitives to Distance Fields to Dynamical Systems](https://arxiv.org/abs/2504.09705). *arXiv preprint*, arXiv:2504.09705.

- 2. Cheng Chi & Zhenjia Xu & Siyuan Feng & Eric Cousineau & Yilun Du & Benjamin Burchfiel & Russ Tedrake & Shuran Song (2024). [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)