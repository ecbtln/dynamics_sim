A generic, extensible simulation library for evolutionary game theory simulations. DyPy provides [Moran](http://en.wikipedia.org/wiki/Moran_process), and [Wright-Fisher](http://en.wikipedia.org/wiki/Genetic_drift#Wright.E2.80.93Fisher_model) processes, as well as [Replicator Dynamics](http://en.wikipedia.org/wiki/Replicator_equation) and makes it simple to execute complex and robust simulations across a range of parameters and visualize the results with beautiful graphs.

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

The ```GameDynamicsWrapper``` and ```VariedGame``` classes take care of simplifying the simulation and graphing processes, and automatically parallelize the computations across all available cores.

To see an example, take a look at the [*Cooperate Without Looking*](https://github.com/ecbtln/cwol_sim/blob/master/cwol.py) subclass along with its associated [simulations](https://github.com/ecbtln/cwol_sim/blob/master/test.py).

####Persistence (coming soon)

DyPy decouples the process of simulating with graphing. This encourages users to run long-running simulations and gather tons of data, and then insert and tweak the graph parameters afterwards.
