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

### Create a Kernel
Define a kernel using GPy Kernels or you can create one for yourself.
[Jupyter Tutorial on GPy Kernels](http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)

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



## A few Words about Composite Kernels

Explain what these tests test and why

```
Give an example
```

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
