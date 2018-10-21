# Contextual-Gaussian-Process-Bandit-Optimization

A straightforward implementation of the CGP-UCB algorithm.[1] 

CGP-UCB is an intuitive upper-confidence style algorithm, in which the payoff function is modeled as a sample from a Gaussian process defined over joint action-context space. It is shown that by mixing and matching kernels for contexts and actions, CGP-UCB can handle a variety of practical applications.[2]

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
# create a set of actions and contexts
actions = np.arange(-3, 3.25, 0.25)
contexts = np.arange(-3, 3.25, 0.25)
# create action-context pairs via a meshgrid.
input_space = np.meshgrid(actions, contexts)
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
[Jupyter Tutorial on GPy Kernels(http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)
[GPy Documentation](https://gpy.readthedocs.io/en/deploy/index.html)
```python
# works on the first dim. of input_space, index=0
kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[0])
# works on the second dim. of input_space, index=1
kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[1])
# composite kernel by additive combination
kernel = kernel1 + kernel2
```

### Initialize and Run
```python
# initialize CGP-UCB
agent = CGPUCB(input_space=input_space, sample_from_environment=environment.sample_noisy, kernel=kernel)
# run for 100 rounds
rounds = 100
  for i in range(rounds):
    agent.learn()
```

## Some Tests and Plots
CGP-UCB is iterated for 300 learning rounds using the DummyEnvironment given above.

### Regret Plot
You can check the source file to see how the regret is plotted. The sublinear regret shows that the CGP-UCB converges to the best action-context pair.
Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Regret Plot")

### Wireframe Plot of Mean and Payoff Function
The following code generates a 3D plot of deduced means at all input-space points and the real environment function.
```python
agent.plot_environment_and_mean()
```
Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Wireframe1")
Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Wireframe2")

### Plotting Slices
You can fix either context or action at some fixed points, here we fix context to [-np.pi/2.0, 0, np.pi/2.0] to produce 3 seperate 2d plots. Following slices are plotted after only 10 rounds of learning.
```python
agent.plot_slices(fixed_dimension=1, slices=[-np.pi/2.0, 0, np.pi/2.0])
```
Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Slices")

### Other Possible Plots
GPy offers a plethora of plotting options. CGP-UCB.gp attribute can be used to access them. A function builtin_plot utilizes CGP-UCB.gp.plot() function with your choice of kwargs. The following code is used to create the contour plot below after 300 rounds.
```python
agent.builtin_plot(projection='2d', title='GPy Contour Plot', xlabel='Actions', ylabel='Contexts')
```
Inline-style: 
![alt text](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization/blob/master/plots/contour_300.png "Contour")

## A few Words about Composite Kernels


## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
