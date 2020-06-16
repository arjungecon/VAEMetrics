# Derivation of VAE with Random Effects

Assume that the DGP is given by
$$
    y_{it} \; = \; X_{it} \beta + \alpha_i + u_{it}
$$
where $u_{it}$ is drawn from a distribution with mean zero and unknown variance $\sigma^2$.

We impose the following assumptions:

- **Strict exogeneity**: $\mathbb{E}[u_{it} \mid \alpha_i, X_{it}]=0$.
- **Distributional assumption on $\alpha_i$**: $f(α_i \mid X_{it}, \beta)$ (Can allow for Correlated Random Effects).
- $\alpha_i$ (and $u_{it}$) are i.i.d. across all $i$ (and all $t$).

We can use Bayes' rule to write that for a given individual $i$,
$$
    \underbrace{f \left( \left\{y_{it} \right\}_{t=1}^T  \, \middle| \, \left\{X_{it} \right\}_{t=1}^T \, ; \, \beta, \sigma^2 \right)}_{p_{\theta}(\mathbf{x}_i)} \; = \; \int \underbrace{f \left( \left\{y_{it} \right\}_{t=1}^T  \, \middle| \, \left\{X_{it} \right\}_{t=1}^T, \alpha_i \, ; \,  \beta, \sigma^2 \right)}_{p_{\theta} (\mathbf{x}_i \mid \mathbf{z}_i)} \underbrace{f\left(\alpha_i \, \middle| \, \left\{X_{it} \right\}_{t=1}^T \, ; \, \beta, \sigma^2  \right)}_{p_{\theta}(\mathbf{z}_i)} \; \text{d} \alpha_i  
$$
For a given training dataset with individuals indexed by $i = 1, \ldots, N$, we have the log-likelihood given by
$$
    \log p_{\theta}(\mathbf{X}) \; =\; \sum_{i = 1}^N \log \int p_{\theta} (\mathbf{x}_i \mid \mathbf{z}_i) \, p_{\theta}(\mathbf{z}_i) \; \text{d} \mathbf{z}_i  
$$

We can also use Bayes' rule to back out the posterior distribution.
$$
    p_{\theta}(\textbf{z} \mid \textbf{X}) \; = \; \frac{p_{\theta} (\mathbf{X} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z}) }{p_{\theta}(\mathbf{X})}
$$

Two issues that arise with a direct inference approach:

- Log-likelihood expression can be analytically intractable and numerically complex for high-dimensional objects.
- Posterior distribution inherits these probabilities.

**Solution of Variational Inference**: Introduce a distribution $q_{\phi}(\mathbf{z})$ that closely approximates $p_{\theta}(\textbf{z} \mid \textbf{x})$. The choice driven by minimizing the KL divergence:
$$
    \operatorname{KL} \left[ q_{\phi}(\mathbf{z}) \| p_{\theta} \left( \mathbf{z} \mid \mathbf{X} \right) \right] \; = \; - \int q_{\phi}(\mathbf{z}) \log \left(\frac{p_{\theta} (\mathbf{z} \mid \mathbf{x})}{q_{\phi}(\mathbf{z})} \right) \; \text{d}\mathbf{z} \; = \; \log p_{\theta}(\mathbf{X}) - \int q_{\phi}(\mathbf{z}) \log \left(\frac{p_{\theta} (\mathbf{z} , \mathbf{x})}{q_{\phi}(\mathbf{z})} \right) \; \text{d}\mathbf{z}
$$