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

def simulate_population_double(func, T_max, M0, R0, beta, u, B1, C2):
    '''
    Apply same logic as above, however, now we have 4 rates, l_x, l_y, mu_x, mu_y
    
    '''
    #initialise time and population
    t = 0
    M = M0 #initialise sensitive cell population
    R = R0 #initialise resistant cell populations
    div = 0
    #store time and population to plot at the end
    times = [0]
    population_s = [M]
    population_r = [R]

    #B = beta * 250

    #store number of cell divisions: birth
    #divisions = [div]
    if isinstance(M, np.ndarray) or isinstance(R, np.ndarray):
        print("M or R has become an array!")

    while t < T_max and (M+R) > 0:

        #extract true rate -> Beta(t)
        l_x, l_y, mu_x, mu_y = func(t, B1, C2)
        #per capita rate
        #should u be included here
        rate_s = l_x + mu_x
        rate_r = l_y + mu_y
        total_rate = (rate_s * M) + (rate_r * R)
        B = beta * (M+R)
        #We generate waiting time to the next event from an exponential distribution with the majorizing rate
        dt = np.random.exponential(1/B)
        t += dt

        U = np.random.uniform()

        # if R > 1e3: 
        # # Set T to T_max so it counts as 'resistant' at the end
        #     t = T_max
        #     R = 1e3 + 1 
        #     break

        #aceept if U <= beta(t)/beta
        if U <= total_rate / B:

            prob_s_birth = l_x * M * (1-u)
            prob_s_death = prob_s_birth + (mu_x * M)
            prob_r_birth = prob_s_death + (l_y * R) + (l_x * M * u)
            prob_r_death = prob_r_birth + (mu_y * R)
            d = np.random.uniform() * total_rate


            if d < prob_s_birth:
                M += 1
            #sensitive cell death
            elif d < prob_s_death:
                M -= 1
            #mutation
            elif d < prob_r_birth:
                R += 1
            else:
                R -= 1
            times.append(t)
            population_s.append(M)
            population_r.append(R)
            #divisions.append(div)

    return np.array(times), np.array(population_s), np.array(population_r)#, np.array(divisions)

def simulate_pulsed_therapy(rates, T_max, M0, beta, u, cycle_length, dosing_length, s):
    '''
    I'm using a different function. It is a copy of the above with some changes
    -------------------------------------
    rates is a dictionary of on/off sets. See below:

    rates = {
        'on':  {'lx': 0.05, 'ly': 0.11, 'mux': 0.1, 'muy': 0.1},
        'off': {'lx': 0.13, 'ly': 0.15, 'mux': 0.1, 'muy': 0.1}
    }
    **Make sure to follow correct naming convention when defining the rates**

    cycle_length is the full length on + off
    dosing_length = just the on portion.

    e.g: 14 day on/off is a 28 day cycle_length and a 14 day dosing_length

    s = initial frequency where,
    S(0) = M(1-s)
    R(0) = Ms

    the remaining arguments are the same as prior
    '''

    t = 0
    M = M0 * (1 - s) #initialise sensitive cell population
    R = M0 * s #initialise resistant cell populations
    
    #store time and population to plot at the end
    times = [0]
    population_s = [M]
    population_r = [R]

    while t < T_max and (M+R) > 0:

        #use modulus operator to check whether dosing schedule is on or off
        if t % cycle_length < dosing_length:
            pulse = 'on'
        else:
            pulse = 'off'
        #now take the set of interest i.e on or off
        r = rates[pulse]
        #extract true rate -> Beta(t)
        l_x = r['lx']
        l_y = r['ly']
        mu_x = r['mux']
        mu_y = r['muy']

        #per capita rate
        #should u be included here
        rate_s = l_x + mu_x
        rate_r = l_y + mu_y
        total_rate = (rate_s * M) + (rate_r * R)
        B = beta * (M+R)
        #We generate waiting time to the next event from an exponential distribution with the majorizing rate
        dt = np.random.exponential(1/B)
        t += dt

        U = np.random.uniform()

        #aceept if U <= beta(t)/beta
        if U <= total_rate / B:

            prob_s_birth = l_x * M * (1-u)
            prob_s_death = prob_s_birth + (mu_x * M)
            prob_r_birth = prob_s_death + (l_y * R) + (l_x * M * u)
            prob_r_death = prob_r_birth + (mu_y * R)
            d = np.random.uniform() * total_rate


            if d < prob_s_birth:
                M += 1
            #sensitive cell death
            elif d < prob_s_death:
                M -= 1
            #mutation
            elif d < prob_r_birth:
                R += 1
            else:
                R -= 1
            times.append(t)
            population_s.append(M)
            population_r.append(R)
            #divisions.append(div)

    return np.array(times), np.array(population_s), np.array(population_r)
def get_concentration(t, D, kappa, tau, loading_dose=False):
    """
    t: current time
    D: dose amount
    kappa: elimination rate (ln(2)/half_life)
    tau: dosing interval (e.g., 1.0 for daily)
    """
    t = float(t)
    n = int(t / tau) + 1  # Number of doses given so far
    time_since_last = t % tau
    
    if loading_dose == True:
        # Immediately at steady-state peak
        c_max = D / (1 - np.exp(-kappa * tau))
    else:
        # Gradual accumulation peak after n doses
        c_max = D * (1 - np.exp(-n * kappa * tau)) / (1 - np.exp(-kappa * tau))
    
    return float(c_max * np.exp(-kappa * time_since_last))

def mean_pulsed_therapy(num_trials, T_max, M0, rates, beta, u, cycle, dosing, s):
    #----------------------
    #Clipped indices is AI
    #----------------------

    # 1. Define the common time grid (e.g., every 1 time unit)
    common_grid = np.linspace(0, T_max, T_max + 1)
    pop_s_trajectories = []
    pop_r_trajectories = []
    #prob_r = []

    r = 0
    for _ in range(num_trials):
        t_sim, S_sim, R_sim = simulate_pulsed_therapy(rates, T_max, M0, beta, u, cycle, dosing, s)
        # if R_sim[-1] > 0:
        #     r += 1
        # else:
        #     r += 0
        # prob_t = r/(_+1)
        # 2. Resample the simulation onto the common grid
        # We use 'searchsorted' to treat it as a step function (constant between events)
        indices = np.searchsorted(t_sim, common_grid, side='right') - 1
        # Handle the case where t=0 is the first index
        indices_s = np.clip(indices, 0, len(S_sim) - 1)
        indices_r = np.clip(indices, 0, len(R_sim) - 1)
        #indices_prob_r = np.clip(indices, 0, len(R_sim) - 1)

        grid_population_s = S_sim[indices_s]
        grid_population_r = R_sim[indices_r]
        #grid_population_prob_r = R_sim[indices_prob_r]

        
        pop_s_trajectories.append(grid_population_s)
        pop_r_trajectories.append(grid_population_r)
        #prob_r.append(prob_t)


    # Convert FIRST, then operate
    pop_s_trajectories = np.array(pop_s_trajectories)
    pop_r_trajectories = np.array(pop_r_trajectories)

    # 3. Calculate Mean (and optionally standard deviation)
    mean_pop_s = np.mean(pop_s_trajectories, axis=0)
    mean_pop_r = np.mean(pop_r_trajectories, axis=0)

    prob_r = np.mean(pop_r_trajectories > 0, axis=0) #probability of resistant cell
    var_r = np.var(pop_r_trajectories , axis=0) #variance of resistant cells

    
    return common_grid, mean_pop_s, mean_pop_r, prob_r, var_r
    
    


def get_mean_trajectory_double(num_trials, T_max, M0, R0, func, beta, u, B1, C2):
    #----------------------
    #Clipped indices is AI
    #----------------------

    # 1. Define the common time grid (e.g., every 1 time unit)
    common_grid = np.linspace(0, T_max, T_max + 1)
    pop_s_trajectories = []
    pop_r_trajectories = []
    #prob_r = []

    r = 0
    for _ in range(num_trials):
        t_sim, S_sim, R_sim = simulate_population_double(func, T_max, M0, R0, beta, u, B1, C2)
        # if R_sim[-1] > 0:
        #     r += 1
        # else:
        #     r += 0
        # prob_t = r/(_+1)
        # 2. Resample the simulation onto the common grid
        # We use 'searchsorted' to treat it as a step function (constant between events)
        indices = np.searchsorted(t_sim, common_grid, side='right') - 1
        # Handle the case where t=0 is the first index
        indices_s = np.clip(indices, 0, len(S_sim) - 1)
        indices_r = np.clip(indices, 0, len(R_sim) - 1)
        #indices_prob_r = np.clip(indices, 0, len(R_sim) - 1)

        grid_population_s = S_sim[indices_s]
        grid_population_r = R_sim[indices_r]
        #grid_population_prob_r = R_sim[indices_prob_r]

        
        pop_s_trajectories.append(grid_population_s)
        pop_r_trajectories.append(grid_population_r)
        #prob_r.append(prob_t)


    # Convert FIRST, then operate
    pop_s_trajectories = np.array(pop_s_trajectories)
    pop_r_trajectories = np.array(pop_r_trajectories)

    # 3. Calculate Mean (and optionally standard deviation)
    mean_pop_s = np.mean(pop_s_trajectories, axis=0)
    mean_pop_r = np.mean(pop_r_trajectories, axis=0)

    prob_r = np.mean(pop_r_trajectories > 0, axis=0) #probability of resistant cell
    var_r = np.var(pop_r_trajectories , axis=0) #variance of resistant cells

    
    return common_grid, mean_pop_s, mean_pop_r, prob_r, var_r

def get_mean_trajectory_single(num_trials, T_max, Z0, func, beta):
    

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

    pop_trajectories = np.array(pop_trajectories)
    

    # 3. Calculate Mean (and optionally standard deviation)
    mean_pop = np.mean(pop_trajectories, axis=0)
    mean_divs = np.mean(div_trajectories, axis=0)

    #prob = np.mean()
    var = np.var(pop_trajectories, axis = 0)
    return common_grid, mean_pop, mean_divs, var
