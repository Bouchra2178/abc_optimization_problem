import random
import numpy as np

def objective_function(x):
    return sum(x**2)

# ABC algorithm implementation
def artificial_bee_colony(obj_func, num_variables, colony_size, num_iterations, lb, ub):
    # Initialization
    best_solution = None
    best_fitness = np.inf
    colony = []
    
    # Generate initial population
    for _ in range(colony_size):
        solution = np.random.uniform(lb, ub, num_variables)
        fitness = obj_func(solution)
        colony.append((solution, fitness))
        
        if fitness < best_fitness:
            best_solution = solution
            best_fitness = fitness
    
    # ABC iterations
    for iteration in range(num_iterations):
        # Employed bees phase
        for i in range(colony_size):
            solution = colony[i][0]
            
            # Select a random solution different from the current one
            while True:
                j = random.randint(0, colony_size - 1)
                if j != i:
                    break
            
            # Generate a new solution by updating a randomly selected variable
            new_solution = solution.copy()
            k = random.randint(0, num_variables - 1)
            phi = random.uniform(-1, 1)
            new_solution[k] = solution[k] + phi * (solution[k] - colony[j][0][k])
            
            # Ensure the new solution is within the search space
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = obj_func(new_solution)
            
            # Greedy selection: update the solution if it improves fitness
            if new_fitness < colony[i][1]:
                colony[i] = (new_solution, new_fitness)
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
        
        # Onlooker bees phase
        # Calculate probabilities for selecting solutions based on their fitness values
        fitness_sum = sum(1 / colony[i][1] for i in range(colony_size))
        probabilities = [1 / (colony[i][1] * fitness_sum) for i in range(colony_size)]
        
        # Select onlooker bees using roulette wheel selection
        for _ in range(colony_size):
            selected_index = np.random.choice(colony_size, p=probabilities)
            selected_solution = colony[selected_index][0]
            
            # Generate a new solution by updating a randomly selected variable
            new_solution = selected_solution.copy()
            k = random.randint(0, num_variables - 1)
            phi = random.uniform(-1, 1)
            new_solution[k] = selected_solution[k] + phi * (selected_solution[k] - best_solution[k])
            
            # Ensure the new solution is within the search space
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = obj_func(new_solution)
            
            # Greedy selection: update the solution if it improves fitness
            if new_fitness < colony[selected_index][1]:
                colony[selected_index] = (new_solution, new_fitness)
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
        
        # Scout bees phase
        for i in range(colony_size):
            if colony[i][1] > best_fitness:
                # Randomly generate a new solution for scout bee
                new_solution = np.random.uniform(lb, ub, num_variables)
                new_fitness = obj_func(new_solution)
                
                colony[i] = (new_solution, new_fitness)
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
    
    return best_solution, best_fitness


# Example usage
num_variables = 5
colony_size = 50
num_iterations = 100
lb = -5.0
ub = 5.0

best_solution, best_fitness = artificial_bee_colony(objective_function, num_variables, colony_size, num_iterations, lb, ub)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)