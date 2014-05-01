A generic, extensible simulation library for evolutionary game theory simulations. DyPy provides [Moran](http://en.wikipedia.org/wiki/Moran_process), and [Wright-Fisher](http://en.wikipedia.org/wiki/Genetic_drift#Wright.E2.80.93Fisher_model) processes, with [Replicator Dynamics](http://en.wikipedia.org/wiki/Replicator_equation) and makes it simple to execute complex and robust simulations across a range of parameters and visualize the results with beautiful graphs.

See documentation [here](http://ecbtln.github.io).

####Requirements

DyPy depends on [matplotlib](http://matplotlib.org) for graphing, and [numpy](http://www.numpy.org) and [joblib](https://pythonhosted.org/joblib/). To install these dependencies, make sure you are in the root directory of the repo and run the following command, which may require sudo.

```bash
$ pip install -r requirements.txt
```

####Usage

The easiest way to get started with DyPy is to subclass the ```Game``` class and define the game that of interest to be simulated by defining its payoff matrix appropriately as a function of various parameters. You can also define a function that classifies equlibria as a function of the distribution of players playing each strategy.

Once the game class is defined, choose a dynamics process and execute the desired simulation. Some options are:

- Simulate a given number of generations of one simulation, and graph the dynamics of each player's strategies over time
- Repeat a given simulation multiple times and return the frequency of each resulting equilibria.
- Vary one or more parameters to the dynamics or game constructors and graph the effect of this variation on the resulting equilibria, either in 2D or 2D graphs.


#### Coming soon



Features
- Parallel
- Generic
- Informative graphs
- Persist changes (coming soon)
- Fully supported range of replicator rules
