# Neural machine translation with Transformer

## Overview

The goal of this repository is to for me to get familiar with a transformer model, which was introduced in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

I base my code on the implementation in the [Annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) blog post. 

## Transformer

Transformer architecture includes 2 main parts: encoder and decoder

<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/transformer.png" align="centre">

###  Encoder
<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/encoder.png" align="centre">

The encoder is a stack of the following blocks:
* Embedding of words ([nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding))
* Positional encoding. My implementation for positional encoding is similar to the one in [Annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) blog post. Please see the blog post for more detail.
* `n_blocks` of the `EncoderBlock` modules.

**Encoder block** 

<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/encoder_block.png" align="centre">

One block of the encoder is a stack of following layers:
  * [nn.LayerNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.LayerNorm) to implement the `Norm` layer in the figure
  * [nn.Dropout](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout) to implement dropout
  * [nn.MultiheadAttention](https://pytorch.org/docs/stable/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention) to implement `Multi-Head Attention`.

* `Feedforward` is simply an MLP processing each position (each element of the source sequence) independently. I used an MLP with:
  * one hidden layer with `n_hidden` neurons
  * a dropout and ReLU activation after the hidden layer
  * an output layer with `n_features` outputs.

* I used dropout in both skip connections of the encoder block.
* In contrast to the [Annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) code, I applied normalization after the skip connection (like it is shown on the figure).

### Decoder
<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/decoder.png" align="centre">

The decoder is a stack of the following blocks:
* Embedding of words ([nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding))
* Positional encoding (`tr.PositionalEncoding` from the attached module)
* `n_blocks` of the `DecoderBlock` modules.
* A linear layer with `tgt_vocab_size` output features.
* Log_softmax nonlinearity.

**Decoder block**
<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/decoder_block.png" align="centre">

One block of the decoder is a stack of following layers:
  * [nn.LayerNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.LayerNorm) to implement the `Norm` layer in the figure
  * [nn.Dropout](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout) to implement dropout
  * [nn.MultiheadAttention](https://pytorch.org/docs/stable/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention) to implement `Multi-Head Attention`.

* `Feedforward` is simply an MLP processing each position (each element of the source sequence) independently. The exact implementation of the MLP is not tested in this notebook. We used an MLP with:
  * one hidden layer with `n_hidden` neurons
  * a dropout and ReLU activation after the hidden layer
  * an output layer with `n_features` outputs.

* I used dropout in both skip connections, similarly to the [Annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

Notes:

The first attention block is self-attention when query, key and value inputs are same. The second attention block uses the encoded `z` values as keys and values, and the outputs of the previous layer as query.

**Subsequent mask**

In the training loop, I will use target sequences (starting with `SOS_token`) as inputs of the decoder. By doing that, we make it possible for the decoder to use previously decoded words when predicting probabilities of the next word. The computations are parallelized in the transformer decoder, and the probabilities of each word in the target sequence are produced by doing one pass through the decoder.

During decoding, we need to make sure that when we compute the probability of the next word, we only use preceding and not subsequent words. In transformers, this is done by providing a mask which tells which elements should be used or ignored when producing the output.

## Model performance on testset

<img src="https://github.com/huongdo108/neural-machine-translation-Transformer/blob/master/images/model_performance.PNG" align="centre">
