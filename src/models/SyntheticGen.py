# contains scripts to generate synthetic data

from models.ModelBuilder import ModelBuilder
from models.Reaction import Reaction
from models.ReactionArchtype import ReactionArchtype
from models.ArchtypeCollections import *
from models.Solver.Solver import Solver
from models.Solver.ScipySolver import ScipySolver
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.Utils import ModelSpecification
from dataclasses import dataclass

from scipy.stats import qmc

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed

def lhs(n_samples, n_features, random_state=None):
    """Latin Hypercube Sampling in [0,1]^d"""
    result = np.empty((n_samples, n_features))
    rng = np.random.default_rng(random_state)
    for i in range(n_features):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(size=n_samples)
        points = cut[:-1] + u * (cut[1] - cut[0])
        rng.shuffle(points)
        result[:, i] = points
    return result


def generate_feature_data_v3(model_spec: ModelSpecification, initial_values: dict, perturbation_type: str, perturbation_params, n, seed=None):
    '''
    Generate a dataframe of perturbed values for the model, version 2, supports lhs sampling
        model_spec: ModelSpecification object
        initial_values: dict, the initial values of the model
        perturbation_type: str, the type of perturbation to apply, either 'uniform', 'gaussian', or 'lhs'
        perturbation_params: dict of parameters for the perturbation, for
            'lhs' perturbation, the params are {'min': float, 'max': float} - lhs is latin hypercube sampling
            'uniform' perturbation, the params are {'min': float, 'max': float}
            'gaussian' perturbation, the param is either {'std': float} or {'rsd': float}
                'rsd' is the relative standard deviation of the perturbation, i.e. std/mean
        n: int, the number of perturbations to generate
        seed: int, the random seed to use for reproducibility
    Returns:
        feature_df: dataframe of the perturbed values
    '''
    if perturbation_type not in ['uniform', 'gaussian', 'lhs']:
        raise ValueError('Perturbation type must be "uniform", "gaussian", or "lhs"')

    if perturbation_type in ['uniform', 'lhs']:
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError('For uniform/lhs perturbation, parameters must contain "min" and "max"')
    elif perturbation_type == 'gaussian':
        if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
            raise ValueError('For gaussian perturbation, parameters must contain "std" or "rsd"')

    all_species = initial_values.keys()

    if perturbation_type == 'lhs':
        min_ = perturbation_params['min']
        max_ = perturbation_params['max']

        # Initialize LHS sampler
        sampler = qmc.LatinHypercube(d=len(all_species), seed=seed)
        lhs_samples = sampler.random(n=n)

        # Scale to [min_, max_] across all dimensions
        scaled_samples = qmc.scale(lhs_samples, [min_] * len(all_species), [max_] * len(all_species))

        # Build dataframe
        feature_df = pd.DataFrame(scaled_samples, columns=all_species)
        return feature_df

    # For uniform and gaussian sampling
    all_perturbed_values = []
    for _ in range(n):
        perturbed_values = {}
        if perturbation_type == 'uniform':
            min_ = perturbation_params['min']
            max_ = perturbation_params['max']
            for s in all_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
        elif perturbation_type == 'gaussian':
            for s in all_species:
                mu = initial_values[s]
                sigma = perturbation_params.get('std', perturbation_params.get('rsd', 0) * mu)
                perturbed_values[s] = np.random.normal(mu, sigma)
        all_perturbed_values.append(perturbed_values)

    feature_df = pd.DataFrame(all_perturbed_values)
    return feature_df



def generate_feature_data_v2(model_spec: ModelSpecification, initial_values: dict, perturbation_type: str, perturbation_params, n, seed=None):
    '''
    Generate a dataframe of perturbed values for the model, version 2, supports lhs sampling
        model_spec: ModelSpecification object
        initial_values: dict, the initial values of the model
        perturbation_type: str, the type of perturbation to apply, either 'uniform', 'gaussian', or 'lhs'
        perturbation_params: dict of parameters for the perturbation, for
            'lhs' perturbation, the params are {'min': float, 'max': float} - lhs is latin hypercube sampling
            'uniform' perturbation, the params are {'min': float, 'max': float}
            'gaussian' perturbation, the param is either {'std': float} or {'rsd': float}
                'rsd' is the relative standard deviation of the perturbation, i.e. std/mean
        n: int, the number of perturbations to generate
        seed: int, the random seed to use for reproducibility
    Returns:
        feature_df: dataframe of the perturbed values
    '''
    if perturbation_type not in ['uniform', 'gaussian', 'lhs']:
        raise ValueError('Perturbation type must be "uniform", "gaussian", or "lhs"')

    if perturbation_type in ['uniform', 'lhs']:
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError('For uniform/lhs perturbation, parameters must contain "min" and "max"')
    elif perturbation_type == 'gaussian':
        if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
            raise ValueError('For gaussian perturbation, parameters must contain "std" or "rsd"')

    all_species = model_spec.A_species + model_spec.B_species

    if perturbation_type == 'lhs':
        min_ = perturbation_params['min']
        max_ = perturbation_params['max']

        # Initialize LHS sampler
        sampler = qmc.LatinHypercube(d=len(all_species), seed=seed)
        lhs_samples = sampler.random(n=n)

        # Scale to [min_, max_] across all dimensions
        scaled_samples = qmc.scale(lhs_samples, [min_] * len(all_species), [max_] * len(all_species))

        # Build dataframe
        feature_df = pd.DataFrame(scaled_samples, columns=all_species)
        return feature_df

    # For uniform and gaussian sampling
    all_perturbed_values = []
    for _ in range(n):
        perturbed_values = {}
        if perturbation_type == 'uniform':
            min_ = perturbation_params['min']
            max_ = perturbation_params['max']
            for s in all_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
        elif perturbation_type == 'gaussian':
            for s in all_species:
                mu = initial_values[s]
                sigma = perturbation_params.get('std', perturbation_params.get('rsd', 0) * mu)
                perturbed_values[s] = np.random.normal(mu, sigma)
        all_perturbed_values.append(perturbed_values)

    feature_df = pd.DataFrame(all_perturbed_values)
    return feature_df

### Generate feature and target data
def generate_feature_data(model_spec: ModelSpecification, initial_values: dict, perturbation_type: str, perturbation_params, n, seed=None):
    '''
    Generate a dataframe of perturbed values for the model
        model_spec: ModelSpecification object   
        model: roadrunner model object
        perturbation_type: str, the type of perturbation to apply, either 'uniform' or 'gaussian'
        perturbation_params: dict of parameters for the perturbation, for
            'lhs' perturbation, the params are {'min': float, 'max': float} - lhs is latin hypercube sampling
            'uniform' perturbation, the params are {'min': float, 'max': float}
            'gaussian' perturbation, the param is either {'std': float} or {'rsd': float}
                'rsd' is the relative standard deviation of the perturbation, i.e. std/mean
        n: int, the number of perturbations to generate
    ''' 
    # validate parameters
    if perturbation_type not in ['uniform', 'gaussian']:
        raise ValueError('Perturbation type must be either "uniform" or "gaussian"')
    
    if perturbation_type == 'uniform':
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError('For uniform perturbation, the parameters must contain "min" and "max" keys')
    elif perturbation_type == 'gaussian':
        # either 'std' or 'rsd' must be in the parameters
        if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
            raise ValueError('For gaussian perturbation, the parameters must contain "std" or "rsd" keys')
    
    # set the random seed if provided, for reproducibility
    if seed is not None:
        np.random.seed(seed)

    all_perturbed_values = []
    all_species = model_spec.A_species + model_spec.B_species
    # if lhs is selected, generate the samples and scale them
    if perturbation_type == 'lhs':
        min_ = perturbation_params['min']
        max_ = perturbation_params['max']
        # Generate LHS samples and scale
        n_features = len(all_species)
        lhs_samples = lhs(n_samples=n, n_features=n_features)
        scaled_samples = min_ + lhs_samples * (max_ - min_)
        feature_df = pd.DataFrame(scaled_samples, columns=all_species)
        return feature_df
    
    for _ in range(n):
        perturbed_values = {}
        if perturbation_type == 'lhs':
            # generate a latin hypercube sample
            min_ = perturbation_params['min']
            max_ = perturbation_params['max']
            for s in model_spec.A_species:
                perturbed_values[s] = np.random.uniform(min_, max_)
            for s in model_spec.B_species:
                perturbed_values[s] = np.random.uniform(min_, max_)
            all_perturbed_values.append(perturbed_values)
        if perturbation_type == 'uniform':
            min_ = perturbation_params['min']
            max_ = perturbation_params['max']  
            for s in model_spec.A_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
            for s in model_spec.B_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
            all_perturbed_values.append(perturbed_values)
        elif perturbation_type == 'gaussian':
            if 'std' in perturbation_params:
                sigma = perturbation_params['std']
            elif 'rsd' in perturbation_params:
                rsd = perturbation_params['rsd']
                sigma = rsd * initial_values[s]
            for s in model_spec.A_species:
                mu = initial_values[s]
                perturbed_values[s] = np.random.normal(mu, sigma)
            for s in model_spec.B_species:
                mu = initial_values[s]
                perturbed_values[s] = np.random.normal(mu, sigma)
            all_perturbed_values.append(perturbed_values)
        
    # put the perturbed values into a dataframe
    feature_df = pd.DataFrame(all_perturbed_values)
    return feature_df

def generate_target_data_diff_spec(model_builds: list[ModelBuilder],
                                   SolverClass: type[Solver],
                                   feature_df: pd.DataFrame,
                                   simulation_params={'start': 0, 'end': 500, 'points': 100},
                                   n_cores=1, verbose=False):
    '''
    Generate the target data for the model, using different model specifications
        model_builds: list of ModelBuilder objects, each object contains a different model specification
        SolverClass: type of Solver, either ScipySolver or RoadrunnerSolver
        feature_df: dataframe of perturbed values
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
    Returns:
        target_df: dataframe of the target values
        time_course_data: list of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    # iterate the dataframe and simulate each perturbation
    all_perturbed_results = []
    time_course_data = []
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]
        # get the model build for this iteration
        model_build = model_builds[i]
        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        solver = SolverClass()
        sbml_str = model_build.get_sbml_model()
        ant_str = model_build.get_antimony_model()
        # check the instance of the solver, if it is RoadrunnerSolver, load the sbml model
        if isinstance(solver, RoadrunnerSolver):
            solver.compile(sbml_str)
        elif isinstance(solver, ScipySolver):
            # if it is ScipySolver, load the antimony model
            solver.compile(ant_str)
        else:
            raise ValueError('For generate_target_data_diff_spec, Solver must be either ScipySolver or RoadrunnerSolver')


        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        # run the simulation, beauty is not only is the solver abstract, has a nice return dataframe format, but 
        # most importantly, it is also stateless, meaning that even though we changed the state values,
        # on the very next call to simulate, it will reset the model to the original state
        res = solver.simulate(start, end, points) 
        # locate the Cp value in the dataframe
        Cp = res['Cp'].iloc[-1]
        time_course = res['Cp'].values
        return Cp, time_course

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        # if n_cores is -1, use all available cores
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_perturbed_results, time_course_data = zip(*results)
        all_perturbed_results = list(all_perturbed_results)
        time_course_data = list(time_course_data)
    else:
        # iterate the dataframe and simulate each perturbation
        all_perturbed_results = []
        time_course_data = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            Cp, time_course = simulate_perturbation(i)
            all_perturbed_results.append(Cp)
            # store the run of Cp into time_course_data
            time_course_data.append(time_course)
            
    target_df = pd.DataFrame(all_perturbed_results, columns=['Cp'])
    return target_df, time_course_data

def generate_target_data_diff_build(model_spec: ModelSpecification, 
                            solver: Solver, 
                            feature_df: pd.DataFrame, 
                            parameter_set: list[dict],
                            simulation_params={'start': 0, 'end': 500, 'points': 100}, 
                            outcome_var='Cp',
                            n_cores=1, verbose=False):
    '''
    Generate the target data for the model, using different parameter sets 
        model_spec: ModelSpecification object   
        solver: Solver object, either ScipySolver or RoadrunnerSolver
        feature_df: dataframe of perturbed values
        parameter_set: list of dicts, each dict contains the parameters to set in the solver
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
    Returns:
        target_df: dataframe of the target values
        time_course_data: list of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    # iterate the dataframe and simulate each perturbation
    all_perturbed_results = []
    time_course_data = []
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]

        params = parameter_set[i]
        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        
        # set the perturbed values into solver 
        solver.set_state_values(perturbed_values)
        # set the parameters into solver
        solver.set_parameter_values(params)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        # run the simulation, beauty is not only is the solver abstract, has a nice return dataframe format, but 
        # most importantly, it is also stateless, meaning that even though we changed the state values,
        # on the very next call to simulate, it will reset the model to the original state
        res = solver.simulate(start, end, points) 
        # locate the Cp value in the dataframe
        Cp = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        return Cp, time_course

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_perturbed_results, time_course_data = zip(*results)
        all_perturbed_results = list(all_perturbed_results)
        time_course_data = list(time_course_data)
    else:
        # iterate the dataframe and simulate each perturbation
        all_perturbed_results = []
        time_course_data = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            try: 
                Cp, time_course = simulate_perturbation(i)
                all_perturbed_results.append(Cp)
                # store the run of Cp into time_course_data
                time_course_data.append(time_course)
            except Exception as e:
                print(f'Error simulating perturbation {i}: {e}')
            
    target_df = pd.DataFrame(all_perturbed_results, columns=[outcome_var])
    return target_df, time_course_data



def generate_target_data(model_spec: ModelSpecification, solver: Solver, feature_df: pd.DataFrame, 
                         simulation_params={'start': 0, 'end': 500, 'points': 100}, 
                         n_cores=1,
                         outcome_var='Cp', 
                         verbose=False):
    '''
    Generate the target data for the model
        model_spec: ModelSpecification object   
        model: roadrunner model object
        feature_df: dataframe of perturbed values
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
    Returns:
        target_df: dataframe of the target values
        time_course_data: list of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    # iterate the dataframe and simulate each perturbation
    all_perturbed_results = []
    time_course_data = []
    
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]

        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        
        # set the perturbed values into solver 
        solver.set_state_values(perturbed_values)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        # run the simulation, beauty is not only is the solver abstract, has a nice return dataframe format, but 
        # most importantly, it is also stateless, meaning that even though we changed the state values,
        # on the very next call to simulate, it will reset the model to the original state
        res = solver.simulate(start, end, points) 
        # locate the Cp value in the dataframe
        Cp = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        return Cp, time_course
    
    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_perturbed_results, time_course_data = zip(*results)
        all_perturbed_results = list(all_perturbed_results)
        time_course_data = list(time_course_data)
    else:
        # iterate the dataframe and simulate each perturbation
        all_perturbed_results = []
        time_course_data = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            # Reset rr model and simulate with each perturbation
            perturbed_values = feature_df.iloc[i]

            # convert the perturbed values to a dictionary
            perturbed_values = perturbed_values.to_dict()
            
            # set the perturbed values into solver 
            solver.set_state_values(perturbed_values)

            # simulate the model and grab only the C and Cp values at the end
            start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
            # run the simulation, beauty is not only is the solver abstract, has a nice return dataframe format, but 
            # most importantly, it is also stateless, meaning that even though we changed the state values,
            # on the very next call to simulate, it will reset the model to the original state
            res = solver.simulate(start, end, points) 
            # locate the Cp value in the dataframe
            Cp = res[outcome_var].iloc[-1]
            all_perturbed_results.append(Cp)
            
            # store the run of Cp into time_course_data
            time_course_data.append(res[outcome_var].values)
            
    target_df = pd.DataFrame(all_perturbed_results, columns=[outcome_var])
    return target_df, time_course_data


def generate_model_timecourse_data_diff_spec(model_builds: list[ModelBuilder], SolverClass: type[Solver], feature_df: pd.DataFrame, simulation_params={'start': 0, 'end': 500, 'points': 100}, capture_species='all', n_cores=1, verbose=False):
    '''
    Generate the time course data for the model, using different model specifications
        model_builds: list of ModelBuilder objects, each object contains a different model specification
        solver: Solver object, either ScipySolver or RoadrunnerSolver
        feature_df: dataframe of perturbed values
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
        capture_species: str or list of str, species to capture in the output, if 'all', captures all species
    Returns:
        output_df: dataframe of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]
        # get the model build for this iteration
        model_build = model_builds[i]
        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        solver = SolverClass()
        sbml_str = model_build.get_sbml_model()
        ant_str = model_build.get_antimony_model()
        # check the instance of the solver, if it is RoadrunnerSolver, load the sbml model
        if isinstance(solver, RoadrunnerSolver):
            solver.compile(sbml_str)
        elif isinstance(solver, ScipySolver):
            # if it is ScipySolver, load the antimony model
            solver.compile(ant_str)
        else:
            raise ValueError('For generate_model_timecourse_data_diff_spec, Solver must be either ScipySolver or RoadrunnerSolver')

        
        # set the perturbed values into solver 
        solver.set_state_values(perturbed_values)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = solver.simulate(start, end, points)
        output = {}
        if capture_species == 'all':
            all_species = model_build.get_state_variables().keys()
            for s in all_species:
                output[s] = res[s].values
        else:
            for s in capture_species:
                output[s] = res[s].values
                sp = s + 'p'
                output[sp] = res[sp].values
        return output
    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_outputs = list(results)
    else:
        # iterate the dataframe and simulate each perturbation
        all_outputs = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                print(f'Error simulating perturbation {i}: {e}')
    output_df = pd.DataFrame(all_outputs)
    # if the output_df is empty, return an empty dataframe
    if output_df.empty:
        raise ValueError('Output dataframe is empty, check the model specifications and feature dataframe')
    # if the output_df is not empty, return the output_df
    return output_df


def generate_model_timecourse_data_diff_build(model_spec: ModelSpecification, solver: Solver, feature_df: pd.DataFrame, parameter_set: list[dict], simulation_params={'start': 0, 'end': 500, 'points': 100}, capture_species='all', n_cores=1, verbose=False):
    '''
    Generate the time course data for the model, using different parameter sets 
        model_spec: ModelSpecification object   
        solver: Solver object, either ScipySolver or RoadrunnerSolver
        feature_df: dataframe of perturbed values
        parameter_set: list of dicts, each dict contains the parameters to set in the solver
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
        capture_species: str or list of str, species to capture in the output, if 'all', captures all species
    Returns:
        output_df: dataframe of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]
        params = parameter_set[i]

        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        
        # set the perturbed values into solver 
        solver.set_state_values(perturbed_values)
        # set the parameters into solver
        solver.set_parameter_values(params)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = solver.simulate(start, end, points) 
        output = {}
        if capture_species == 'all':
            all_species = model_spec.A_species + model_spec.B_species + model_spec.C_species
            for s in all_species:
                output[s] = res[s].values
                sp = s + 'p'
                output[sp] = res[sp].values
        else:
            for s in capture_species:
                output[s] = res[s].values
                sp = s + 'p'
                output[sp] = res[sp].values
        return output

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_outputs = list(results)
    else:
        # iterate the dataframe and simulate each perturbation
        all_outputs = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                print(f'Error simulating perturbation {i}: {e}')

    output_df = pd.DataFrame(all_outputs)
    return output_df

def generate_model_timecourse_data(model_spec: ModelSpecification, 
                                   solver: Solver, 
                                   feature_df: pd.DataFrame, 
                                   simulation_params={'start': 0, 'end': 500, 'points': 100}, 
                                   capture_species='all', 
                                   n_cores=1, verbose=False):
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError(
            'Simulation parameters must contain "start", "end" and "points" keys')

    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]

        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()

        # set the perturbed values into solver
        solver.set_state_values(perturbed_values)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = solver.simulate(start, end, points)
        output = {}
        if capture_species == 'all':
            all_species = model_spec.A_species + model_spec.B_species + model_spec.C_species
            for s in all_species:
                output[s] = res[s].values
                sp = s + 'p'
                output[sp] = res[sp].values
        else:
            for s in capture_species:
                output[s] = res[s].values
                sp = s + 'p'
                output[sp] = res[sp].values
        return output

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_outputs = list(results)
    else:
        # iterate the dataframe and simulate each perturbation
        all_outputs = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            output = {}

            # Reset rr model and simulate with each perturbation
            perturbed_values = feature_df.iloc[i]

            # convert the perturbed values to a dictionary
            perturbed_values = perturbed_values.to_dict()

            # set the perturbed values into solver
            solver.set_state_values(perturbed_values)

            # simulate the model and grab only the C and Cp values at the end
            start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
            res = solver.simulate(start, end, points)
            if capture_species == 'all':
                all_species = model_spec.A_species + model_spec.B_species + model_spec.C_species
                for s in all_species:
                    output[s] = res[s].values
                    sp = s + 'p'
                    output[sp] = res[sp].values
            else:
                for s in capture_species:
                    output[s] = res[s].values
                    sp = s + 'p'
                    output[sp] = res[sp].values
            all_outputs.append(output)

    output_df = pd.DataFrame(all_outputs)
    return output_df


def generate_model_timecourse_data_v3(all_species: dict,
                                   solver: Solver, 
                                   feature_df: pd.DataFrame, 
                                   simulation_params={'start': 0, 'end': 500, 'points': 100}, 
                                   n_cores=1, verbose=False):
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError(
            'Simulation parameters must contain "start", "end" and "points" keys')

    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]

        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()

        # set the perturbed values into solver
        solver.set_state_values(perturbed_values)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = solver.simulate(start, end, points)
        output = {}
        for s in all_species.keys():
            output[s] = res[s].values
        return output

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_outputs = list(results)
    else:
        # iterate the dataframe and simulate each perturbation
        all_outputs = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            output = {}

            # Reset rr model and simulate with each perturbation
            perturbed_values = feature_df.iloc[i]

            # convert the perturbed values to a dictionary
            perturbed_values = perturbed_values.to_dict()

            # set the perturbed values into solver
            solver.set_state_values(perturbed_values)

            # simulate the model and grab only the C and Cp values at the end
            start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
            res = solver.simulate(start, end, points)
            for s in all_species.keys():
                output[s] = res[s].values
            all_outputs.append(output)

    output_df = pd.DataFrame(all_outputs)
    return output_df


def generate_model_timecourse_data_diff_build_v3(all_species: dict, 
                                                 solver: Solver, 
                                                 feature_df: pd.DataFrame, 
                                                 parameter_set: list[dict], 
                                                 simulation_params={'start': 0, 'end': 500, 'points': 100}, 
                                                 capture_species='all', n_cores=1, verbose=False):
    '''
    Generate the time course data for the model, using different parameter sets 
        model_spec: ModelSpecification object   
        solver: Solver object, either ScipySolver or RoadrunnerSolver
        feature_df: dataframe of perturbed values
        parameter_set: list of dicts, each dict contains the parameters to set in the solver
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
        capture_species: str or list of str, species to capture in the output, if 'all', captures all species
    Returns:
        output_df: dataframe of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    def simulate_perturbation(i):
        # Reset rr model and simulate with each perturbation
        perturbed_values = feature_df.iloc[i]
        params = parameter_set[i]

        # convert the perturbed values to a dictionary
        perturbed_values = perturbed_values.to_dict()
        
        # set the perturbed values into solver 
        solver.set_state_values(perturbed_values)
        # set the parameters into solver
        solver.set_parameter_values(params)

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = solver.simulate(start, end, points) 
        output = {}
        for s in all_species.keys():
            output[s] = res[s].values
        return output

    # use parallel processing to speed up the simulation
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(delayed(simulate_perturbation)(i) for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose))
        all_outputs = list(results)
    else:
        # iterate the dataframe and simulate each perturbation
        all_outputs = []
        # iterate the dataframe and simulate each perturbation
        for i in tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                print(f'Error simulating perturbation {i}: {e}')

    output_df = pd.DataFrame(all_outputs)
    return output_df


