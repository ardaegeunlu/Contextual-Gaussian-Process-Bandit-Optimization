# coding: utf-8

import GPy
from mpl_toolkits.mplot3d import Axes3D
GPy.plotting.change_plotting_library('matplotlib')
import matplotlib.pylab as plt
import numpy as np

"""
For more details about the algorithm please check the following papers.
"Information Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting", Srinivas et al., 2012.
"Contextual Gaussian Process Bandit Optimization", Krause et al., 2011.
"""

class CGPUCB(object):

  def __init__(self, input_space, sample_from_environment, kernel, delta=0.80):

    """
    :param input_space: The input space of possible values that can be played.
    :param sample_from_environment: sampling function which returns observed value.
    :param delta: delta needs to be in interval (0,1) exclusively. Delta is a hyper-parameter that is used to choose
    beta, where beta decides the trade-off between exploitation and exploration. A higher beta means more exploration,
    and a smaller beta results in greedier choices. Delta and beta are inversely correlated.
    """
    self.input_mesh = np.array(input_space)
    self.sample_from_environment = sample_from_environment
    self.beta = None
    self.delta = delta

    self.input_space = self.input_mesh.reshape(self.input_mesh.shape[0], -1).T
    self.input_space_size = self.input_space.size
    self.mu = np.array([0. for _ in range(self.input_space.shape[0])])
    self.sigma = np.array([0.5 for _ in range(self.input_space.shape[0])])

    if kernel is None:
      # default kernel
      kernel = GPy.kern.RBF(input_dim=self.input_space[0].size, variance=1., lengthscale=1.)

    self.kernel = kernel
    self.kernel.fix()  # the kernel parameters will NOT be optimized.
    self.gp = None

    self.X = []
    self.Y = []
    self.round = 1

  def cgp_ucb_rule(self):
    """
    this point selection strategy combines a greedy of choice of choosing a point with high mu, together with
    an exploratory choice of choosing a point with high variance; achieving a balance of exploration & exploitation.
    :return: next point to be sampled.
    """
    return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

  def learn(self):
    """
    1 iteration of the learning algorithm. Iterate this for T rounds of your choice.
    :return:
    """
    # choose optimal beta.
    self.beta = self.optimal_beta_selection(self.round, self.input_space_size, self.delta)
    # choose new sampling point using cgp-ucb.
    grid_idx = self.cgp_ucb_rule()
    self.sample(self.input_space[grid_idx])
    # fit the data to a gaussian process.
    self.gp = GPy.models.GPRegression(np.array(self.X), np.array(self.Y), self.kernel)
    self.gp.optimize(messages=False)
    # get mu and sigma predictions.
    self.mu, variances = self.gp.predict(self.input_space)
    self.sigma = np.sqrt(variances)
    # increment round #.
    self.round += 1

  def sample(self, x):
    """
    :param x: the point to be sampled from environment.
    :return:
    """
    y = self.sample_from_environment(x)
    self.X.append(x)
    self.Y.append(y)

  def builtin_plot(self, **kwargs):
    """
    note that GPy offers a plethora of plotting options, they can always be used with self.gp to suit many
    different needs.
    :return:
    """
    self.gp.plot(**kwargs)

  def plot_environment_and_mean(self):
    """
    plot the noisy_environment function, the deduced mean and the data points on a 3d-wireframe scatter plot.
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('Environment, Mean and Data on a 3d Plot')
    ax.set_xlabel('Action')
    ax.set_ylabel('Context')
    ax.set_zlabel('Reward')

    # plot self.mu on a wire frame.
    ax.plot_wireframe(self.input_mesh[0], self.input_mesh[1],
                      self.mu.reshape(self.input_mesh[0].shape), alpha=0.6, color='g', label='mean')
    # now plot sample environment at all values of the input space and plot.
    ax.plot_wireframe(self.input_mesh[0], self.input_mesh[1],
                      np.array(self.sample_from_environment(self.input_mesh)).reshape(self.input_mesh[0].shape),
                      alpha=0.5, color='b', label='environment')
    # scatter plot data on top.
    ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.Y, c='r',
               marker='o', alpha=1.0, label='data')
    ax.legend()

  def plot_slices(self, fixed_dimension, slices, density=False):
    """
    In a contextual bandit, you can fix a context or action and plot the unfixed dimension vs confidence and mean
    by specifying the fixed dimension and what values you would like to fix it to.
    :param fixed_dimension: 0 or 1.
    :param slices: The value or an array-like values which fixed_dimension will be fixed to.
    :return:
    """
    if fixed_dimension is 0:
      fixed_dim_name = 'action'
      xlabel = 'context'
    else:
      fixed_dim_name = 'context'
      xlabel = 'action'
    number_of_slices = slices.__len__()
    figure = GPy.plotting.plotting_library().figure(number_of_slices, 1)

    for i, y in zip(range(number_of_slices), slices):
      self.gp.plot(figure=figure,plot_density=density, fixed_inputs=[(fixed_dimension, y)], row=(i + 1),
                   plot_data=False, title='slice at %s=%f' % (fixed_dim_name, y), xlabel=xlabel, ylabel='reward')

  def optimal_beta_selection(self, t, input_space_size, delta):
    """
    :param t: the current round t.
    :param input_space_size: |D| of input space D.
    :param delta: hyperparameter delta where 0 < delta < 1, exclusively.
    :return: optimal beta for exploration_exploitation trade-off at round t.
    """
    return 2 * np.log(input_space_size * (t ** 2) * (np.pi ** 2) / (6 * delta))


### End of CGP-UCB implementation. Rest of the code is used for testing.

class DummyEnvironment(object):

  def sample(self, x):
    return np.sin(x[0]) + np.cos(x[1])

  def sample_noisy(self, x):
    return [self.sample(x) + np.random.normal(loc=0.0, scale=0.02)]

  def find_best_input(self, input_space):
    return np.argmax(self.sample(input_space))



def plot_regret(best, agent):

  plt.figure(0)

  cum_best = np.cumsum(np.array(best))
  plt.plot(cum_best, label="best strategy reward")

  cum_agent = np.cumsum(np.array(agent))
  plt.plot(cum_agent, label="agent reward")

  cum_regret = cum_best - cum_agent
  plt.plot(cum_regret, label="regret")

  plt.xlabel("rounds")
  plt.ylabel("cumulative rewards/regret")

  plt.title("Regret and Cumulative Rewards")
  plt.legend()
  plt.show()

  return


if __name__ == '__main__':

  # define input space
  actions = np.arange(-3, 3.25, 0.25)
  contexts = np.arange(-3, 3.25, 0.25)
  # create action-context pairs via a meshgrid.
  input_space = np.meshgrid(actions, contexts)

  # create an environment
  environment = DummyEnvironment()

  # define a kernel using GPy Kernels or you can create one for yourself.
  # Jupyter Tutorial on Kernels:
  # -> http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb
  # GPy Documentation:
  # -> https://gpy.readthedocs.io/en/deploy/index.html

  # works on the first column of D, index=0
  kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[0])
  # works on the second column of D, index=1
  kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1., active_dims=[1])
  # composite kernel
  kernel = kernel1 + kernel2

  # initialize CGP-UCB
  agent = CGPUCB(input_space=input_space, sample_from_environment=environment.sample_noisy, kernel=kernel)

  rounds = 100
  for i in range(rounds):
    if i % 100 == 0:
      print("@round%d" % (i))
    agent.learn()

  # print(agent.gp)
  agent.plot_environment_and_mean()
  agent.plot_slices(fixed_dimension=1, slices=[-np.pi/2.0, 0, np.pi/2.0])
  agent.builtin_plot(projection='2d', title='GPy Contour Plot', xlabel='Actions', ylabel='Contexts')


  best_strategy = agent.input_space[environment.find_best_input(input_space=np.array(input_space))]
  plot_regret(best=np.full(fill_value=environment.sample(best_strategy), shape=rounds), agent=np.array(agent.Y))
  plt.show(block=True)



