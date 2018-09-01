import numpy as np

class ES(object):
    def __init__(self, env, simulate_tlim=200, bound=[-50,50], generations = 300, population_size=100, offspring_size=50, render=False):
        self.gen = generations
        self.bound = bound
        self.pop_n = population_size
        self.kid_n = offspring_size
        self.actions = max(1,np.prod(env.action_space.shape))
        action_sample = env.action_space.sample()
        self.scaler_action = type(action_sample)==int or type(action_sample)==float
        self.action_dtype = type(action_sample)
        self.states = np.prod(env.observation_space.shape)
        self.env = env
        self.percetron_shape = (self.actions, self.states)
        self.dna_n = self.actions*self.states
        self.render = render
        self.tlim = simulate_tlim
    def initialization(self, mean=0.0, std_dev=2.0, std_dev_m=1.0):
        pop = dict(DNA=std_dev * (mean + np.random.randn(self.pop_n, self.dna_n)),
               mut_strength=std_dev_m * np.random.randn(self.pop_n, self.dna_n))
        return pop
    def get_fitness(self, pop, test_mode=False):
        scores = []
        for idv in pop:
            Wxb = idv.reshape(self.percetron_shape) # shape: (actions, states)
            tot_reward = 0
            observation = self.env.reset().ravel()
            done = False
            step_lim = self.tlim
            while not done and step_lim>0:
                if self.render:
                    self.env.render()
                action = np.sign(Wxb @ observation) # percetron
                if self.scaler_action and self.action_dtype==int:
                    action = np.clip(self.action_dtype(action), 0, 1)
                observation, reward, done, info = self.env.step(action)
                observation = observation.ravel()
                tot_reward += reward
                if not test_mode:
                    step_lim -= 1
            scores.append(tot_reward)
        return np.asarray(scores)
    def get_offspring(self,pop):
        kids = {'DNA': np.empty((self.kid_n, self.dna_n))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for kv, ks in zip(kids['DNA'], kids['mut_strength']):
            p1, p2 = np.random.choice(np.arange(self.pop_n), size=2, replace=False)
            cp = np.random.randint(0, 2, self.dna_n, dtype=np.bool)  # crossover points
            kv[cp] = pop['DNA'][p1, cp]
            kv[~cp] = pop['DNA'][p2, ~cp]
            ks[cp] = pop['mut_strength'][p1, cp]
            ks[~cp] = pop['mut_strength'][p2, ~cp]

            # mutate (change DNA based on normal distribution)
            ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
            kv += ks * np.random.randn(*kv.shape)
            kv[:] = np.clip(kv, *self.bound)    # clip the mutated value
        return kids
    def put_kids(self, pop, kids):
        for key in ['DNA', 'mut_strength']:
            pop[key] = np.append(pop[key], kids[key], axis=0)
        return pop
    def selection(self, pop):
        fitness = self.get_fitness(pop['DNA'])
        good_idx = np.argsort(fitness)[-self.pop_n:]
        fitness = fitness[good_idx]
        pop['mut_strength'] = pop['mut_strength'][good_idx]
        pop['DNA'] = pop['DNA'][good_idx]
        return fitness, pop
    def fit(self, initial_pop, runs=1, show_progress=False):
        for r in range(runs):
            pop = initial_pop.copy()
            for f in range(self.gen):
                kids = self.get_offspring(pop)
                pop  = self.put_kids(pop, kids)
                fitness, pop  = self.selection(pop)
                if show_progress:
                    print('[%d/%d] | [%d/%d] | mean fitness: %.2f | best fitness: %.2f'%(r+1,runs,f+1,self.gen,fitness.mean(), fitness.max()))
        return fitness.max(), pop['DNA'][-1]
