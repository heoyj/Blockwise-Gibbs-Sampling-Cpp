# C++ Project 

## Block-wise Gibbs Sampling using 

## Abstract

To make a parsimonious model when initially having a large number of predictors, it is inevitable to identify the best subset of such predictors. In this case, a spike-and-slab prior facilitates an efficient Gibbs sampling by shrinking the parameter estimates with the mixture of two prior distributions. To implement far more efficient sampling, we introduced a block-wise updating approach to the beta coefficient. In the end, we found that our Gibbs sampling code reduced the running time’s of general Gibbs sampling. Also, it was faster than the Gibbs sampling without the block-wise updating approach approximately two- thirds(2/3).

## Usage

Block-wise Gibbs sampling can be useful when there are a lot of covariates. This blockwise Gibbs sampling method is from a model, called spike and slab model (eq.(2)) in <a href="https://arxiv.org/pdf/math/0505633.pdf" target="_blank">Spike and Slab Variable Selection: Frequentist and Bayesian Strategies.</a>






