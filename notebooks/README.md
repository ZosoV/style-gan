# Reproduction: Image2StyleGAN

The following reproduction was performed to the Eastern European Machine Learning Summer School 2021, "Deep Learning and Reinforcement Learning".

The reproduced article is [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](https://arxiv.org/abs/1904.03189)

This reproduction consists of two notebooks where you can find the source code together with a detailed explanation of each part.

1. [embedding_algorithm.ipynb](embedding_algorithm.ipynb): Notebook with the implementation of the Image2StyleGAN algorithm. Also, the code was partially edited for include further experimentation, studying the use of initializer and perceptual losses.
2. [semantic_editing.ipynb](semantic_editing.ipynb): Notebook with the implementation of three semantic editing operators: morphing, style transfer and expression transfer.

Additional notebooks:

1. [embedding_algorithm_simple.ipynb](embedding_algorithm_simple.ipynb): This is a reduced version of [embedding_algorithm.ipynb](embedding_algorithm.ipynb) where you can find only the original algorithm Image2StyleGAN without editions.
2. [improved_embedding.ipynb](improved_embedding.ipynb): This is a improved version of the algorithm Image2StyleGAN, also called ImprovedImage2StyleGAN (II2S). It adds a regularizer term during the optimization steps, based on an affine transformation implemented with PCA and a new latent space representation P-space. The original paper could be found here [Improved StyleGAN Embedding: Where are the Good Latents?](https://arxiv.org/abs/2012.09036)