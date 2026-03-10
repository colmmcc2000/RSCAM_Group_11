import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate

def simulate_population_single(func, T_max, Z0, beta):
    '''
    Thinning Method:
    In stochastic modelling we say events occur independently and randomly with an averge rate. This is a poisson process.
    This is a single-type non-homogenous poisson process
    Property of poisoon process is waiting times between events follow an exponetial distribution.
    Where R is the average rate, witing time is 1/R.
    However, in this case we do not know average rate between events as rates are time dependent.
    We use a majorising rate beta, where beta is always >= true rate.
    beta >= R so 1/beta is always smaller. We sample more events (over-sample), but select occurence with proabability (true/rate / beta).
    This is thinning. 
    The remainder of the code is simple enough. If an event occurs we check probability of birth vs death and select one.
    Store population and time at each step. Go until T_max or population goes extinct.
    We can also store cell divisions each time a birth occurs.

    -----------------------------------------------------
    func = [lambda(t), mu(t)]
    T_max = Max length of time given population may grow unbounded
    Z0 = initial population
    beta = majorising rate. should be max rate + a bit. Fromonline it seems to be max rate + a bit, but I'm unsure why.
        It may be to make sure that there's always a chance the event does not occur if true rate is max
    
    '''
    #initialise time and population
    t = 0
    Z = Z0
    div = 0
    #store time and population to plot at the end
    times = [0]
    population = [Z]

    #B = beta * 250

    #store number of cell divisions: birth
    divisions = [div]
    

    while t < T_max and Z > 0:

        #extract true rate -> Beta(t)
        l, mu = func(t)
        #per capita rate
        total_rate = (l + mu) * Z
        B = beta * Z
        #We generate waiting time to the next event from an exponential distribution with the majorizing rate
        dt = np.random.exponential(1/B)
        t += dt

        U = np.random.uniform()

        #aceept if U <= beta(t)/beta
        if U <= total_rate / B:
            if np.random.uniform() < l / (l + mu):
                Z += 1
                div += 1
            else:
                Z -= 1
            times.append(t)
            population.append(Z)
            divisions.append(div)

    return np.array(times), np.array(population), np.array(divisions)


def get_mean_trajectory(num_trials, T_max, Z0, func, beta):
    #----------------------
    #This is a lot of gemini. Need to understand how to correct arrays to a single length where they vary for plot
    #----------------------

    # 1. Define the common time grid (e.g., every 1 time unit)
    common_grid = np.linspace(0, T_max, T_max + 1)
    pop_trajectories = []
    div_trajectories = []

    for _ in range(num_trials):
        t_sim, X_sim, div_sim = simulate_population_single(func, T_max, Z0, beta)
        
        # 2. Resample the simulation onto the common grid
        # We use 'searchsorted' to treat it as a step function (constant between events)
        indices = np.searchsorted(t_sim, common_grid, side='right') - 1
        # Handle the case where t=0 is the first index
        indices = np.clip(indices, 0, len(X_sim) - 1)
        grid_population = X_sim[indices]
        grid_divisions = div_sim[indices]
        
        pop_trajectories.append(grid_population)
        div_trajectories.append(grid_divisions)

    # 3. Calculate Mean (and optionally standard deviation)
    mean_pop = np.mean(pop_trajectories, axis=0)
    mean_divs = np.mean(div_trajectories, axis=0)
    return common_grid, mean_pop, mean_divs
