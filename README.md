<div align="center"><img src="https://raw.githubusercontent.com/chainer/chainer/master/docs/image/chainer_red_h.png" width="400"/></div>

# Chainer: A deep learning framework
[![pypi](https://img.shields.io/pypi/v/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![GitHub license](https://img.shields.io/github/license/chainer/chainer.svg)](https://github.com/chainer/chainer)
[![travis](https://img.shields.io/travis/chainer/chainer/master.svg)](https://travis-ci.org/chainer/chainer)
[![coveralls](https://img.shields.io/coveralls/chainer/chainer.svg)](https://coveralls.io/github/chainer/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)

[**Website**](https://chainer.org/)
| [**Docs**](https://docs.chainer.org/en/stable/)
| [**Install Guide**](https://docs.chainer.org/en/stable/install.html)
| **Tutorials** ([ja](https://tutorials.chainer.org/ja/))
| **Examples** ([Official](https://github.com/chainer/chainer/tree/master/examples), [External](https://github.com/chainer-community/awesome-chainer))
| [**Concepts**](https://docs.chainer.org/en/stable/guides/)
| [**ChainerX**](#chainerx)

**Forum** ([en](https://groups.google.com/forum/#!forum/chainer), [ja](https://groups.google.com/forum/#!forum/chainer-jp))
| **Slack invitation** ([en](https://bit.ly/join-chainer-slack), [ja](https://bit.ly/join-chainer-jp-slack))
| **Twitter** ([en](https://twitter.com/ChainerOfficial), [ja](https://twitter.com/ChainerJP))

# My Contributions (Aniket Didolkar)

## Before GSoC selection

PR | Description | Merged? |
-- | ----------- | ------- |
[#6472](https://github.com/chainer/chainer/pull/6472) | Implementation of sigmoid for ChainerX | yes |
[#6476](https://github.com/chainer/chainer/pull/6476) | Dot Product for higher dimensions for ChainerX | yes |
[#6496](https://github.com/chainer/chainer/pull/6496) | Elementwise power operator for ChainerX | yes |
[#6715](https://github.com/chainer/chainer/pull/6715) | Implementation of absolute for ChainerX | yes |
[#6731](https://github.com/chainer/chainer/pull/6731) | Implementation of Relu for ChainerX | yes |

## After GSoC selection

### Main Feature-Based PRs
These PRs include the main features that were to be implemented during GSoC.

PR | Description | Docs | Merged? |
-- | ----------- | ---- | ------ |
[#7764](https://github.com/chainer/chainer/pull/7764) | This PR includes the CUDNN and CPU implementation of `LSTM/BiLSTM`, `GRU/BiGRU` and `RNN/BiRNN` | [link](https://docs.chainer.org/en/latest/chainerx/reference/routines.html#rnn) | yes |
[#7783](https://github.com/chainer/chainer/pull/7783) | This includes the implementation of `S-LSTM` routine as an activation function | [link](https://docs.chainer.org/en/latest/chainerx/reference/generated/chainerx.slstm.html#chainerx.slstm) | yes |
[#7720](https://github.com/chainer/chainer/pull/7720) | This includes the implementation of `TreeLstm` as an activation function |[link](https://docs.chainer.org/en/latest/chainerx/reference/generated/chainerx.tree_lstm.html#chainerx.tree_lstm)| yes | 
[#7784](https://github.com/chainer/chainer/pull/7784) | This includes the implementation of word embeddings | | no |

### Supplementary PRs
These are the supplementary PRs that were implemented that were necessary for merging the feature-based PRs mentioned above. These mainly include
* Chainer test fixes so that ChainerX routines could be tested from Chainer.
* Documentation fixes for the features above

PR | Description | Merged? |
-- | ----------- | ------ |
[#7804](https://github.com/chainer/chainer/pull/7804) | Simplify `n_step_rnn/birnn` test in Chainer | yes | 
[#7806](https://github.com/chainer/chainer/pull/7806) | Simplify `n_step_gru/bigru` test in Chainer | yes |
[#7807](https://github.com/chainer/chainer/pull/7807) | Simplify `n_step_lstm/bilstm` test in Chainer | yes |
[#7805](https://github.com/chainer/chainer/pull/7805) | Simplify `slstm` test in Chainer | yes |
[#7808](https://github.com/chainer/chainer/pull/7808) | Simplify `lstm` (as an activation function) test in Chainer | yes |
[#7881](https://github.com/chainer/chainer/pull/7881) | Simplify `TreeLSTM` test in Chainer | yes | 
[#7903](https://github.com/chainer/chainer/pull/7903) | Simplify word embedding test in Chainer | yes |
[#7985](https://github.com/chainer/chainer/pull/7985) | Fix RNN docs for ChainerX | yes | 

### Short Description of GSoC project
The main aim of this project was to implement all the recurrent neural network based routines available in Chainer (under [chainer.function.rnn](https://github.com/chainer/chainer/tree/master/chainer/functions/rnn)) in ChainerX. All the feature based PRs that have been merged can be called using the C++ and the Python API of ChainerX. For each of the [chainer rnn functions](https://github.com/chainer/chainer/tree/master/chainer/functions/rnn), I have implemented the `forward_chainerx` function, which means the ChainerX rnn functions can be used by calling the Chainer rnn functions and specifying the appropriate device as mentioned [here](https://docs.chainer.org/en/stable/chainerx/tutorial/index.html#run-your-chainer-code-with-chainerx).

For each feature the basic procedure was as follows : 
* Implement the feature using the other routines already available if possible.
* If [CUDNN](https://developer.nvidia.com/cudnn) supports the given feature, then implement the forward and backward CUDNN kernels for the given feature.
* Write tests for the given feature.
* Write documentation for the given feature.

Most of the Supplementary PRs involve test fixes. When implementing the `forward_chainerx`, there had to be a way to test the ChainerX code from Chainer. The original tests were not able to do that. Hence, I had to modify the tests such that the ChainerX code could be tested seemlessly from Chainer.

### Future Work
* Support [Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) in the `n_step_lstm/bilstm`, `n_step_gru/bigru` and `n_step_rnn/birnn` routines.
* Demonstrate through an example, the speed-up that ChainerX offers over Chainer and PyTorch in the case of RNNs.

-------------------------------------------------------------------------------------------------------------------

*Chainer* is a Python-based deep learning framework aiming at flexibility.
It provides automatic differentiation APIs based on the **define-by-run** approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks.
It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.
For more details about Chainer, see the documents and resources listed above and join the community in Forum, Slack, and Twitter.



  
  

## Stable version

The stable version of current Chainer is separated in here: [v5](https://github.com/chainer/chainer/tree/v5).

## Installation

To install Chainer, use `pip`.

```sh
$ pip install chainer
```

To enable CUDA support, [set up CUDA](https://docs.nvidia.com/cuda/index.html#installation-guides) and install [CuPy](https://github.com/cupy/cupy).

```sh
$ pip install cupy
```

[See the installation guide for more details](https://docs.chainer.org/en/stable/install.html).


## Docker image

We are providing the official Docker image.
This image supports [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Login to the environment with the following command, and run the Python interpreter to use Chainer with CUDA and cuDNN support.

```
$ nvidia-docker run -it chainer/chainer /bin/bash
```


## Contribution

Any contributions to Chainer are welcome!
If you want to file an issue or send a pull request, [please follow the contribution guide](https://docs.chainer.org/en/stable/contribution.html).


## ChainerX

See the [ChainerX documentation](https://docs.chainer.org/en/stable/chainerx/index.html).


## License

MIT License (see `LICENSE` file).


## More information

- [Release notes](https://github.com/chainer/chainer/releases)


## Reference

Tokui, S., Oono, K., Hido, S. and Clayton, J.,
Chainer: a Next-Generation Open Source Framework for Deep Learning,
*Proceedings of Workshop on Machine Learning Systems(LearningSys) in
The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, (2015)
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)


Akiba, T., Fukuda, K. and Suzuki, S.,
ChainerMN: Scalable Distributed Deep Learning Framework,
*Proceedings of Workshop on ML Systems in
The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017)
[URL](http://learningsys.org/nips17/assets/papers/paper_25.pdf), [BibTex](chainermn_bibtex.txt)
