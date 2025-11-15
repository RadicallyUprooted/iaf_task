# IAFStep implementation details

### `flip` (bool)
Flips the inputs to alternate the dependency when stacking multiple IAF steps.

### `forward(self, x)`
1. First, we take a batch of samples `x` from the base distribution.
2. We split it into `x1` and `x2`. However, we only need `x1` here.
3. Since `m1` and `s1` are known, we compute `m2` and `s2` by passing `x1` through `self.ar_net`.
4. Concatenate them to get `m` and `s`.
5. Apply the transformation from Eq.14.

### `reverse(self, y)`
1. We take a batch of samples `y` from the data distribution.
2. Similarly, we split it into `y1` and `y2`. Only now we would need them both.
3. Using the independent parameters `m1` and `s1`, we first compute `x1` using the inverse transform for Eq.14: `x1 = (y1 - (1 - sigma1) * m1) / sigma1`.
4. Now we can compute `m2` and `s2` by passing `x1` through `self.ar_net`.
5. Compute `x2` using the inverse transform for Eq.14 the similar way: `x2 = (y2 - (1 - sigma2) * m2) / sigma2`.
6. Finally, return concatenated `x`.

### `log_determinant(self, ...)`
Similar to the BatchNorm code presented, we use the cached `s` parameters to compute the sum of logsigmoids of them. 

# Training results
<p align="center">
  <img src="training_animation.gif" alt="Training Animation " width="512" height ="512">
</p>

# Training details

The task description mentions that we can achieve optimal convergence using around 10 [IAFStep, BatchNorm] transformations, where IAFStep uses a plain feedforward NN (dense -> relu -> dense -> ...) with 16 hidden neurons. However, I doubted every statement in the description and experimented with different setups. In the end, I achieved approximately similar results for every experiment done. I decided to stick with the following setup:
1. `self.ar_net`: dense(1, 64) -> relu -> dense(64, 64) -> relu -> dense(64, 2). But it doesn't make that much difference if we use 16 hidden neurons instead, to be honest.
2. `self.s1_param` is initialized as 1.0, since it is mentioned in the paper that training is more stable in this case (2.0 is also a valid choice).
3. Custom initialization of the output layer for `self.ar_net` (which is `m2` and `s2`) matters in case when we don't use the invertible BatchNorm in the transforms, so I left it commented out.
4. `layers = 12` with flips included.
5. `torch.optim.AdamW(..., lr=5e-3, weight_decay=0)` seems to be a more modern choice. I tried different `weight_decay` values, which of course affect the training, but decided to stick with zero. Also note that when `weight_decay=0`, AdamW is equivalent to Adam.
6. `torch.optim.lr_scheduler.ExponentialLR(..., gamma=0.9998)` for the decay to happen a bit sooner, since we start with a relatively high learning rate.
7. `batch_size=2048`.

# Additional notes

1. `from sklearn.mixture.gaussian_mixture import ...` is deprecated, it is now `sklearn.mixture._gaussian_mixture`.
2. In the invertible BatchNorm implementation, the `.mean()` is redundant, since we already get a scalar at this point.
3. `StandardNormalDistribution` returns a scalar in the `logprob()` method. It then broadcasts this scalar inside the `TransformedDistribution`, which seems kinda incorrect to me, so I made a little change to avoid unnecessary broadcasting. The only necessary broadcasting here happens when we include the log-det of the BatchNorm transform.
