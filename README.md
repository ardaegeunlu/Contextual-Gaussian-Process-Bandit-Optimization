# Contextual-Gaussian-Process-Bandit-Optimization

A straightforward implementation of the CGP-UCB algorithm [1]. 

CGP-UCB is an intuitive upper-confidence style algorithm, in which the payoff function is modeled as a sample from a Gaussian process defined over joint action-context space. It is shown that by mixing and matching kernels for contexts and actions, CGP-UCB can handle a variety of practical applications [2].

## Dependencies

You need to have the following libraries.
```
GPy
matplotlib
numpy
```
## How to Run

### Define your Input Space
```python
# create arrays of actions and contexts
actions = np.arange(-3, 3.25, 0.25)
contexts = np.arange(-3, 3.25, 0.25)
```

### Create an Environment
```python
environment = DummyEnvironment()
```
DummyEnvironment used for testing is given below.
```python
class DummyEnvironment(object):
  def sample(self, x):
    return np.sin(x[0]) + np.cos(x[1])
  def sample_noisy(self, x):
    return [self.sample(x) + np.random.normal(loc=0.0, scale=0.02)]
```

### Create a Kernel
Define a kernel using GPy Kernels or you can create one for yourself.
[Jupyter Tutorial on GPy Kernels](http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)
[GPy Documentation](https://gpy.readthedocs.io/en/deploy/index.html)
```python
# works on the first dim. of input_space, index=0
kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[0])
# works on the second dim. of input_space, index=1
kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[1])
# composite kernel by the product of two kernels.
kernel = kernel1 * kernel2
```

### Initialize and Run
```python
# initialize CGP-UCB
agent = CGPUCB(actions=actions, contexts=contexts, sample_from_environment=environment.sample_noisy, kernel=kernel)
rounds = 300
number_of_contexts = contexts.size
best_strategy_rewards = []
for i in range(rounds):
  # choose a random context.
  context_index = np.floor(np.random.rand()*number_of_contexts)
  # iterate learning algorithm for 1 round.
  agent.learn(context_index)
  # save best_strategy's reward for the current context. (Only used to plot regret.)
  best_strategy_rewards.append(environment.get_best_reward_in_context(context_index))
```

## Some Tests and Plots
CGP-UCB is iterated for 300 learning rounds using the DummyEnvironment given above.
### Regret Plot
The following code plots the regret. Note that plot_regret is not a part of the CGP-UCB class.
```python
plot_regret(best=best_strategy_rewards, agent=agent.Y)
```
The sublinear regret shows that the CGP-UCB converges to the best action-context pair. 
![](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/regret_300.png "Regret Plot")

### Wireframe Plot of Mean and Payoff Function
The following code generates a 3D plot of deduced means at all input-space points and the real environment function. 
```python
agent.plot_environment_and_mean()
``` 
![](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/wireframe_300_1.png "Wireframe1")

![](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/wireframe_300_2.png "Wireframe2")

### Plotting Slices
You can fix either context or action at some fixed points, here we fix context to [-np.pi/2.0, 0, np.pi/2.0] to produce 3 seperate 2d plots. 
```python
agent.plot_slices(fixed_dimension=1, slices=[-np.pi/2.0, 0, np.pi/2.0])
```
![](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/slices_300.png "Slices")

### Other Possible Plots
GPy offers a plethora of plotting options. CGP-UCB.gp attribute can be used to access them. The function "builtin_plot()" utilizes CGP-UCB.gp.plot() function with your choice of kwargs. The following code is used to create the contour plot below after 300 rounds.
```python
agent.builtin_plot(projection='2d', title='GPy Contour Plot', xlabel='Actions', ylabel='Contexts')
```
![](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/contour_300.png "Contour")

## A few Words about Composite Kernels
### Additive Kernel
An alternative is to consider the additive combination (kS ⊕ kZ) of kernels. The intuition behind this construction is that a GP with additive kernel can be understood as a generative model, which first samples a function fS(s, z) that is constant along z, and various along s with regularity as expressed by ks; it then samples a function fz(s, z), which varies along z and is constant along s; then f = fs + fz. Thus, the fz component models overall trends according to the context (e.g., encoding assumptions about similarity within clusters of contexts), and the fS models action-specific deviation from this trend [3].
### Product Kernel
The intuition behind this product kernel is a conjunction of the notions of similarities induced by the kernels over context and action spaces: Two context-action pairs are similar (large correlation) if the contexts are similar and actions are similar. Note that many kernel functions used in practice are already in product form. For example, if kZ and kS are squared
exponential kernels, then the product k = kZ ⊗ kS is a squared exponential kernel [4].


## Acknowledgments
### Reference
[1,2,3,4] Krause, A., Ong, C.S. (2011). Contextual Gaussian Process Bandit Optimization. Advances in Neural Information Processing Systems 24 (NIPS 2011), pp.2447-2455.
### Also check for implementation
Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2012). Information-Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting. IEEE Transactions on Information Theory, 58(5), pp.3250-3265.

Thanks to https://github.com/tushuhei/gpucb for initial template on GP-UCB.
