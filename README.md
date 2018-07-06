This repository contains code and data used to generate the results in
[Learning Qualitatively Diverse and Interpretable Rules for Classification](https://arxiv.org/abs/1806.08716).

## Main Idea

### Interpretable + Interpretable = Uninterpretable

Let's start with a thought experiment about interpretability and ambiguity in classification, which we'll introduce with a story from [David Barber](https://www.amazon.com/Bayesian-Reasoning-Machine-Learning-Barber/dp/0521518148):

> A father decides to teach his young son what a sports car is.  Finding it
> difficult to explain in words, he decides to give some examples. They stand on
> a motorway bridge and as each car passes underneath, the father cries out
> "that’s a sports car!" when a sports car passes by. After ten minutes, the
> father asks his son if he's understood what a sports car is. The son says,
> "sure, it’s easy". An old red VW Beetle passes by, and the son shouts –
> "that’s a sports car!". Dejected, the father asks – "why do you say that?".
> "Because all sports cars are red!", replies the son.

In this story, we have a training dataset (`x` = cars, `y` = cries), that can
be classified in two qualitatively different ways (`way 1` = car shape, or
whatever the dad is thinking, and `way 2` = the color red). These ways will
generalize differently on new data, even though they make the same predictions
on the training dataset. One nice thing about both ways, however, is that they
are easy to explain and mentally simulate, which we'll say makes them
"interpretable." If we could be sure that our training procedure would return
a model that exactly matched `way 1` or `way 2`, then even if we got the one
which wouldn't generalize, at least we could understand it, given the right
visualization or explanation tools (which in this story, correspond to the
conversation the father and son have at the end).

But what if we didn't learn either `way 1` or `way 2`, but a mixture of both?
After all, _any_ such mixture would be perfectly accurate on the training set,
so there's nothing disincentivizing us from learning it.  If this mixture was a
simple linear combination whose weights didn't change for different inputs,
then life might remain simple; if we saw a new car that was red but not
sportscar-shaped, then one of the two ways would just win, and the same one
would win every time. The model would remain easy to explain and mentally
simulate. But if the weights changed depending on other aspects of the input
(e.g. if the son switched between `way 1` and `way 2` based on
how sunny it was, except for Toyotas), then it would start becoming
significantly harder to explain and mentally simulate.

The worry behind this project is that this is exactly what tends to happen with
almost every type of machine learning model on ambiguous datasets (and we'll present
some basic results to that effect, but also
[see](http://web.mit.edu/torralba/www/iccv2001.pdf)
[these](https://arxiv.org/abs/1803.09797)
[citations](https://arxiv.org/abs/1711.11561)).
Given multiple interpretable options, we inevitably learn all of them, which
means we learn none of them; and even with the most transparent model class, or
most well-designed explanation technique, the actual _function_ that we learn
will be challenging for humans to understand.

### Learning Independent Classifiers

How can we start trying to solve this problem? One option is to try to learn a
model that's somehow _sparse_; perhaps not in its sensitivity to input
features, but to latent generative or causal factors (if we have a
representation that can disentangle them). However, (1) it's hard to perfectly
disentangle such factors, (2) it's hard to achieve sparsity for expressive
model classes like neural networks, and (3) this only gives us one model.

In this project, our goal is to overcome dataset ambiguity by learning an _ensemble_ of
classifiers that make statistically independent predictions, but nevertheless
all perform perfectly on the training set. These two goals are clearly at
odds, because if all models perfectly classify the training set, then their
predictions can't be independent. But independence is always relative to a
distribution, and our training set might not be distributed in the same way as
our test set, or as we think the model may be used in the real world; really, we
would like to learn an ensemble that _extrapolates_ outside our training set in
independent ways.

Because we don't know this distribution over which we want independence, we
instead settle for _local_ independence, which we define as follows: for a
given input `x`, our classifiers (which might predict the same label for `x`)
nevertheless have independent probability outputs when we perturb `x` (with
infinitesimal-variance Gaussian noise). It turns out that this condition is
satisfied if and only if our classifiers have orthogonal _input gradients_ at
`x`. So in this work, we try to achieve local independence everywhere by
jointly training an ensemble of models with a penalty on the cosine similarity
of their gradients. And we find that on datasets with ground-truth
interpretable classification functions, this technique lets us learn models
that recover them. For more details, check out the
[paper](https://arxiv.org/abs/1806.08716) or the notebooks below!

## Repository Structure

- [2D Illustrative Examples](./2D-Illustrative-Examples.ipynb) contains code and data to replicate our 2D experiments, which illustrate our method and serve as a simple litmus test for diverse ensemble training methods.
- [8D Feature Selection](./8D-Feature-Selection.ipynb) replicates our feature selection experiment, which shows that our method can solve this important case and also demonstrates how training many model classes normally leads to a dense, unintuitive combination of individually interpretable functions.
- [dSprites Latent Space](./dSprites-Latent-Space.ipynb) replicates our results on [dSprites](https://github.com/deepmind/dsprites-dataset), a 64x64 image dataset, which demonstrates that our method can still be used in high-dimensional settings where input features aren't individually meaningful.

## Citation

```
@inproceedings{ross2018learning,
  author    = {Ross, Andrew and Pan, Weiwei and Doshi-Velez, Finale},
  title     = {Learning Qualitatively Diverse and Interpretable Rules for Classification},
  booktitle = {2018 ICML Workshop on Human Interpretability in Machine Learning (WHI 2018)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1806.08716},
}
```
