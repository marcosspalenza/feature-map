import scipy as sp
import numpy as np
from sklearn.metrics import pairwise
import random

"""
Based in hello_evolve.py implementation
Made by Colin Drake [2011]
Link: https://gist.github.com/cfdrake/973505
Under BSD Licence
"""
class Simple_GA:
    def __init__(self, start_indiv, pop_size, n_generations, tfidf):
        self.start_indiv     = start_indiv
        self.individual_size    = np.shape(self.start_indiv)[1]
        self.pop_size    = pop_size
        self.n_generations = n_generations
        self.tfidf = tfidf

    def executeGA(self):

        population = self.random_population()

        for generation in range(self.n_generations):

            weighted_population = []

            for individual in population:
                fitness_val = self.fitness(individual)

                if fitness_val == 0:
                    pair = (individual, 2.0)
                else:
                    pair = (individual, fitness_val)

                weighted_population.append(pair)

            population = []

            for _ in range(int(self.pop_size/2)):
                ind1 = self.selection(weighted_population)
                ind2 = self.selection(weighted_population)

                ind1, ind2 = self.crossover(ind1, ind2)

                population.append(self.mutate(ind1))
                population.append(self.mutate(ind2))


        fittest = population[0]
        minimum_fitness = self.fitness(population[0])
        
        for individual in population:
            ind_fitness = self.fitness(individual)
            if ind_fitness < minimum_fitness:
                fittest = individual
                minimum_fitness = ind_fitness

        return fittest
    
    def selection(self, items):

        value = random.randint(0,(self.pop_size-1))
        n1 = items[value]
        value = random.randint(0,(self.pop_size-1))
        n2 = items[value]
        if n1[1] < n2[1]:
            return n1[0]
        else:
            return n2[0]

    def random_population(self):
        pop = []
        pop.append(self.start_indiv.tolist()[0])
        for i in range(self.pop_size-1):
            pop.append(np.random.randint(2, size=self.individual_size).tolist())
        return pop

    def fitness(self, dna):
        xInd = yInd = range(len(dna))
        newIndividual = sp.sparse.csr_matrix((dna, (xInd, yInd)))
        newFeatures = np.multiply(self.tfidf, newIndividual)
        centroid = newFeatures.mean(axis=0)
        mean_sim = np.mean([pairwise.pairwise_distances(newFeatures[x], centroid, metric="cosine") for x in range(np.shape(newFeatures)[0])])
        non_zeros = newIndividual.sum()/np.shape(newFeatures)[1]
        result = non_zeros/mean_sim
        return result

    def mutate(self, dna):
        mutation_chance = 50
        rand_pos = np.random.randint(self.individual_size-1, size=int((self.individual_size-1)/100))
        for pos in rand_pos:
            if int(random.random()*mutation_chance) == 1:
                if dna[pos] == 1:
                    dna[pos] = 0
                else:  
                    dna[pos] = 1
        return dna

    def crossover(self, dna1, dna2):
        pos = int(random.random()*(self.individual_size))
        arr1 = dna1[:pos]+dna2[pos:]
        arr2 =  dna2[:pos]+dna1[pos:]
        return (arr1, arr2)