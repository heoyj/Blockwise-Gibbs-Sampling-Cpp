# C++ Project 

## Block-wise Gibbs Sampling using 

## Abstract

To make a parsimonious model when initially having a large number of predictors, it is inevitable to identify the best subset of such predictors. In this case, a spike-and-slab prior facilitates an efficient Gibbs sampling by shrinking the parameter estimates with the mixture of two prior distributions. To implement far more efficient sampling, we introduced a block-wise updating approach to the beta coefficient. In the end, we found that our Gibbs sampling code reduced the running time’s of general Gibbs sampling. Also, it was faster than the Gibbs sampling without the block-wise updating approach approximately two- thirds(2/3).

## 1. Usage

Block-wise Gibbs sampling can be useful when there are a lot of covariates. This blockwise Gibbs sampling method is from a model, called spike and slab model (eq.(2)) in <a href="https://arxiv.org/pdf/math/0505633.pdf" target="_blank">Spike and Slab Variable Selection: Frequentist and Bayesian Strategies.</a>

From this underlying model, the source code follows the updating scheme from SVS Gibbs sampler in Ishwaran1 in main function. The blockwise update of beta coefficients is executed at first for every iteration from betaUpdate. Once the coefficients are up- dated, the covariance part of beta are updated from phiUpdate, tauUpdate, wUpdate, sigmaUpdate and gammaUpdate, sequentially. Burning and thinning are automatically set up from authors and the user doesn’t need to specify them.

## 2. Variables

### 2.1. Input variables

1. Dataset : .csv formatted with header of covariate names. The first column is Y (re- sponse variable) and the rest of columns are X (covariates) values. We recommend you to try using the data with enormous amount of covariates since it is on our pur- pose of making block-wise updating gibbs sampler. This dataset should be attached as an ”argv[1]”.

2. Block size : When K is the number of covariates, we have K corresponding beta coefficients. When the user specify the block size p, then our code updates the beta coefficients by p-size of beta coefficients. The user can set p as any number between 1 and K including 1 and K. But the smaller p is, the slower the code execution takes.

3. Parameters of prior distribution : The users can use the prior parameters that they want. If they want to use their own priors, they can answer ”y” in the question of asking such. Otherwise, our program will automatically use our own defined parameters for the prior distributions. The following parameters need to be specified by users: the prior precision matrix tau_k^2, the prior precision matrix sigma^2, the zero-like value nu_0, the initial prior sigma^2, the initial weight omega.

### 2.2. Output variables

The output variables are estimates of beta coefficient. Therefore, the parameter estimates are K parameters which includes our X covariates and one intercept. The example of parameter estimates are the follows.

| Parameters | Last draw | Posterior Mean | 2.5th percentile | Convergence | 
| :--------: | :-------: | :------------: | :--------------: | :---------: |
| beta 0     |    0.3    | 0.2            | -0.2             | 0.6         |
| beta 1     |           |                |                  |             |









