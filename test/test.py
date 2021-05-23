import torch
import pyro
import pyro.distributions as dist

#pyro.set_rng_seed(101)


def scale(guess):
    # A priori guess ( i.e theoretical calculation)
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    print(weight)
    measurement = pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)
    return measurement


print(scale(guess=9.0))