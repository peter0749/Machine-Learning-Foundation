import numpy as np
from es_base import ES
import gym

GEN = 10
POP = 100
OFF = 100
RUNS = 1
TLIM = 25000
REN = False # render result

env = gym.make('CartPole-v1')
env = env.unwrapped

es = ES(env, simulate_tlim=TLIM, generations=GEN, population_size=POP, offspring_size=OFF, render=REN)
pop = es.initialization()
best_fitness, best_idv = es.fit(pop, RUNS, True)

print(best_fitness)
print(best_idv)
np.save('CartPole-v1-perceptron-model.npy', best_idv.reshape(es.percetron_shape))

es.render=True
es.get_fitness(best_idv[np.newaxis], test_mode=True)

