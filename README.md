# nlp_models
various and sundry on NLP

This repo contains my work on NLP models intending to get my head around them. It starts with the classic Encoder of the Transformer model in [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf). There are many Keras implementation of the Encoder available open source: e.g. the one on the official Keras example page by [Apoorv Nandan](https://keras.io/examples/nlp/text_classification_with_transformer/) or another one listed among the [Tensorflow tutorials](https://www.tensorflow.org/tutorials/text/transformer). Both do a great job but also suffer from some incompletness -- at least from my point of view. The first does not include masking, the second is not written as succinctly as possible in Tensorflow 2.+. Both implementations are missing dynamic padding discussed by [Michaël Benesty](https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e) which reduces training time considerable because training of an Transformer-Encoder grows quadratically with the sequence length. The present implementation tackels these issues:
* introducing a padding mask for the encoder,
* write the model in a Tensorflow 2.+ style,
* introducing dynamic padding.

The code itself is accompanied by an Ipython Notebook where we train our model against Keras' IMBD dataset.
