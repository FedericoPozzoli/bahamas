"""
This module implements ERYN transdimensional sampler for Bayesian analysis with variable number of EGP nodes.
It provides functionality for reversible-jump MCMC sampling where the number of EGP knots can vary.

Functions:
    ERYNInference: Main class for setting up and running ERYN transdimensional sampling.
    LogLikeERYN: Log-likelihood wrapper compatible with ERYN's interface.

Dependencies:
    - eryn
    - JAX
    - NumPyro
    - psd_function (custom module)
"""
from bahamas.psd_strain import psd_function as psd
from bahamas.logger_config import logger

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.moves import GaussianMove, DistributionGenerateRJ


def whittle_lik(coords, data, freqs, response, sources, dt, t1, t2, dof, gen2, inds=None):
    """
    ERYN-compatible log-likelihood wrapper.
    
    Converts ERYN's (coords, inds) format to sources dictionary, then calls model_psd.
    
    Parameters:
    -----------
    coords : dict
        Dictionary of coordinate arrays: {branch_name: array[nleaves, ndim]}
    data : list
        Observed data segments
    freqs : list
        Frequency grids for each segment
    response : list
        Response functions for each segment
    sources : dict
        Source configuration (needed for parameter mapping)
    dt : float
        Time step
    t1, t2 : array
        Start and end times for each segment
    dof : array
        Degrees of freedom for each segment
    gen2 : bool
        Generation 2 flag
    inds : dict, optional
    
    Returns:
    --------
    log_likelihood : float
        Log-likelihood value using Whittle approximation
    """
    # Convert ERYN format to sources dictionary
    sources_dict = _eryn_coords_to_sources(coords, inds, sources)
    
    # Compute log-likelihood using Whittle approximation
    log_likelihood = 0.0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f = np.array(freqs[j])
            psd_model = psd.model_psd(
                freqs=f, 
                response=response[j][i], 
                sources=sources_dict, 
                t1=t1[j], 
                t2=t2[j], 
                tdi=i, 
                gen2=gen2
            )
            log_likelihood += (
                -0.5 * np.sum((np.abs(tdi) ** 2) / psd_model) 
                - 0.5 * len(f) * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(psd_model))
            )
    
    return log_likelihood

def gamma_lik(coords, data, freqs, response, sources, dt, t1, t2, dof, gen2, inds = None):
    """
    ERYN-compatible log-likelihood wrapper.
    
    Converts ERYN's (coords, inds) format to sources dictionary, then calls model_psd.
    
    Parameters:
    -----------
    coords : dict
        Dictionary of coordinate arrays: {branch_name: array[nleaves, ndim]}
    inds : dict
        Dictionary of boolean indicators: {branch_name: array[nleaves]}
    data : list
        Observed data segments
    freqs : list
        Frequency grids for each segment
    response : list
        Response functions for each segment
    sources : dict
        Source configuration (needed for parameter mapping)
    dt : float
        Time step
    t1, t2 : array
        Start and end times for each segment
    dof : array
        Degrees of freedom for each segment
    gen2 : bool
        Generation 2 flag
    inds : dict, optional
    
    Returns:
    --------
    log_likelihood : float
        Log-likelihood value using Gamma approximation
    """
    # Convert ERYN format to sources dictionary
    sources_dict = _eryn_coords_to_sources(coords, inds, sources)
    
    # Compute log-likelihood using Gamma approximation
    log_likelihood = 0.0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f = np.array(freqs[j])
            psd_model = psd.model_psd(
                freqs=f, 
                response=response[j][i], 
                sources=sources_dict, 
                t1=t1[j], 
                t2=t2[j], 
                tdi=i, 
                gen2=gen2
            )
            log_likelihood += jnp.sum(
                (dof[j] / 2 - 1) * jnp.log(jnp.abs(tdi) ** 2) 
                - (dof[j] / 2) * jnp.log(psd_model) 
                - jsc.special.gammaln(dof[j] / 2) 
                - (jnp.abs(tdi) ** 2) / (2 * psd_model)
            )
    
    return log_likelihood


#def _eryn_coords_to_sources(coords, inds, sources_config):
#    """
#    Convert ERYN (coords, inds) format to sources dictionary for model_psd.
#    
#    Parameters:
#    -----------
#    coords : list of arrays
#        List of coordinate arrays, one per branch: [branch0_coords, branch1_coords, ..., egp_coords]
#        Each array has shape [nwalkers, nleaves, ndim]
#    inds : list of arrays
#        List of boolean indicator arrays: [branch0_inds, branch1_inds, ..., egp_inds]
#        Each array has shape [nwalkers, nleaves]
#    sources_config : dict
#        Source configuration from sources.yaml
#    
#    Returns:
#    --------
#    sources_dict : dict
#        {source_name: {param_name: value, ...}}
#    """
#    sources_dict = {}
#    
#    # Get branch names (excluding 'egp')
#    branch_names = [name for name in sources_config.keys() if name != 'egp']
#    
#    # Find which source has EGP enabled
#    egp_source = None
#    for source_name, param_list in sources_config.items():
#        for param in param_list:
#            if param.get('egp', False):
#                egp_source = source_name
#                break
#        if egp_source:
#            break
#    
#    # Process scalar branches
#    for branch_idx, branch_name in enumerate(branch_names):
#        param_list = sources_config[branch_name]
#        
#        # Get coords and inds for this branch
#        # coords is a list: [branch0, branch1, ..., egp]
#        branch_coords = coords[branch_idx]  # Shape: typically [1, ndim] for scalar branches
#        branch_inds = inds[branch_idx] if inds is not None else None
#        
#        # Handle different shapes
#        if branch_coords.ndim == 2:
#            # Shape: [nleaves, ndim]
#            # For scalar branches, use first (and only) leaf
#            if branch_inds is not None:
#                active_mask = branch_inds
#                if not np.any(active_mask):
#                    logger.warning(f"No active leaves for branch '{branch_name}'")
#                    continue
#                active_coords = branch_coords[active_mask][0]
#            else:
#                active_coords = branch_coords[0]
#        elif branch_coords.ndim == 1:
#            # Shape: [ndim]
#            active_coords = branch_coords
#        else:
#            logger.error(f"Unexpected coords shape for branch '{branch_name}': {branch_coords.shape}")
#            continue
#        
#        # Map coordinates to parameter names
#        branch_dict = {}
#        param_idx = 0
#        for param in param_list:
#            if param.get('bounds') is not None:
#                # Sampled parameter
#                branch_dict[param['name']] = float(active_coords[param_idx])
#                param_idx += 1
#            else:
#                # Fixed parameter
#                if 'injected' in param:
#                    branch_dict[param['name']] = param['injected']
#                elif 'default' in param:
#                    branch_dict[param['name']] = param['default']
#        
#        sources_dict[branch_name] = branch_dict
#    
#    # Attach EGP parameters to the designated source
#    # EGP branch is the last one in coords list
#    egp_coords = coords[-1]  # Last element is EGP
#    egp_inds = inds[-1] if inds is not None else None
#    
#    if egp_source and egp_source in sources_dict:
#        # Handle different EGP coord shapes
#        if egp_coords.ndim == 2:
#            # Shape: [nleaves, 1]
#            if egp_inds is not None:
#                active_egp = egp_coords[egp_inds]  # Get active nodes
#            else:
#                active_egp = egp_coords
#        elif egp_coords.ndim == 1:
#            # Shape: [nleaves,]
#            if egp_inds is not None:
#                active_egp = egp_coords[egp_inds]
#            else:
#                active_egp = egp_coords
#        else:
#            logger.error(f"Unexpected EGP coords shape: {egp_coords.shape}")
#            active_egp = np.array([])
#        
#        # Add delta parameters
#        if active_egp.ndim == 2:
#            # Shape: [n_active, 1]
#            for i, delta_value in enumerate(active_egp[:, 0]):
#                sources_dict[egp_source][f'delta_{i}'] = float(delta_value)
#        else:
#            # Shape: [n_active,]
#            for i, delta_value in enumerate(active_egp):
#                sources_dict[egp_source][f'delta_{i}'] = float(delta_value)
#        
#        # Add frequency bounds
#        for param in sources_config[egp_source]:
#            if param.get('egp', False):
#                sources_dict[egp_source]['fmin'] = param.get('fmin', 1e-5)
#                sources_dict[egp_source]['fmax'] = param.get('fmax', 1e-1)
#                break
#    
#    return sources_dict

def _eryn_coords_to_sources(coords, inds, sources_config):
    """
    Convert ERYN (coords, inds) format to sources dictionary for model_psd.
    
    Parameters:
    -----------
    coords : dict or list/array
        Coordinates in ERYN format - can be dict {branch_name: array} or list [branch0, branch1, ...]
    inds : dict or list/array or None
        Boolean indicators - can be dict {branch_name: array} or list [branch0, branch1, ...]
    sources_config : dict
        Source configuration from sources.yaml
    
    Returns:
    --------
    sources_dict : dict
        {source_name: {param_name: value, ...}}
    """
    sources_dict = {}
    
    # Get branch names (excluding 'egp')
    branch_names = [name for name in sources_config.keys() if name != 'egp']
    all_branches = branch_names + ['egp']
    
    # Convert coords and inds to dictionary format if they're lists/arrays
    if not isinstance(coords, dict):
        # coords is a list/array: [branch0, branch1, ..., egp]
        coords = {name: coords[i] for i, name in enumerate(all_branches)}
    
    if inds is not None and not isinstance(inds, dict):
        # inds is a list/array: [branch0, branch1, ..., egp]
        inds = {name: inds[i] for i, name in enumerate(all_branches)}
    
    # Find which source has EGP enabled
    egp_source = None
    for source_name, param_list in sources_config.items():
        for param in param_list:
            if param.get('egp', False):
                egp_source = source_name
                break
        if egp_source:
            break
    
    # Process scalar branches
    for branch_name in branch_names:
        param_list = sources_config[branch_name]
        
        # Check if branch exists in coords
        if branch_name not in coords:
            logger.warning(f"Branch '{branch_name}' not found in coords")
            continue
            
        branch_coords = coords[branch_name]
        branch_inds = inds.get(branch_name) if inds is not None else None
        
        # Handle None coords
        if branch_coords is None:
            logger.warning(f"Branch '{branch_name}' has None coords")
            continue
        
        # Handle different shapes
        if branch_coords.ndim == 2:
            # Shape: [nleaves, ndim]
            if branch_inds is not None:
                active_mask = branch_inds
                if not np.any(active_mask):
                    logger.warning(f"No active leaves for branch '{branch_name}'")
                    continue
                active_coords = branch_coords[active_mask][0]
            else:
                active_coords = branch_coords[0]
        elif branch_coords.ndim == 1:
            # Shape: [ndim]
            active_coords = branch_coords
        else:
            logger.error(f"Unexpected coords shape for branch '{branch_name}': {branch_coords.shape}")
            continue
        
        # Map coordinates to parameter names
        branch_dict = {}
        param_idx = 0
        for param in param_list:
            if param.get('bounds') is not None:
                # Sampled parameter
                branch_dict[param['name']] = float(active_coords[param_idx])
                param_idx += 1
            else:
                # Fixed parameter
                if 'injected' in param:
                    branch_dict[param['name']] = param['injected']
                elif 'default' in param:
                    branch_dict[param['name']] = param['default']
        
        sources_dict[branch_name] = branch_dict
    
    # Attach EGP parameters to the designated source
    if 'egp' not in coords:
        # No EGP branch in coords - return without EGP parameters
        return sources_dict
        
    egp_coords = coords['egp']
    egp_inds = inds.get('egp') if inds is not None else None
    
    if egp_source and egp_source in sources_dict:
        # Handle None or empty EGP coords
        if egp_coords is None:
            active_egp = np.array([])
        elif egp_coords.ndim == 2:
            # Shape: [nleaves, 1]
            if egp_inds is not None:
                active_egp = egp_coords[egp_inds]
            else:
                active_egp = egp_coords
        elif egp_coords.ndim == 1:
            # Shape: [nleaves,]
            if egp_inds is not None:
                active_egp = egp_coords[egp_inds]
            else:
                active_egp = egp_coords
        else:
            logger.error(f"Unexpected EGP coords shape: {egp_coords.shape}")
            active_egp = np.array([])
        
        # Add delta parameters only if we have active nodes
        if active_egp.size > 0:
            if active_egp.ndim == 2:
                # Shape: [n_active, 1]
                for i, delta_value in enumerate(active_egp[:, 0]):
                    sources_dict[egp_source][f'delta_{i}'] = float(delta_value)
            elif active_egp.ndim == 1:
                # Shape: [n_active,]
                for i, delta_value in enumerate(active_egp):
                    sources_dict[egp_source][f'delta_{i}'] = float(delta_value)
        
        # Add frequency bounds
        for param in sources_config[egp_source]:
            if param.get('egp', False):
                sources_dict[egp_source]['fmin'] = param.get('fmin', 1e-5)
                sources_dict[egp_source]['fmax'] = param.get('fmax', 1e-1)
                break
    
    return sources_dict

class ERYNInference:
    """
    Main class for setting up and running ERYN transdimensional inference.
    """
    
    def __init__(self, log_like_func, config, **kwargs):
        """
        Initialize ERYN inference.
        
        Parameters:
        - log_like_func: Base likelihood function (not used directly, we use LogLikeERYN)
        - config (dict): Configuration dictionary.
        - data, freqs, response, sources, dt, t1, t2, dof, gen2: Standard BAHAMAS parameters.
        """
        # Store sources config - needed for coords to sources conversion
        self.sources = kwargs.get('sources', None)
        if self.sources is None:
            raise ValueError("sources configuration must be provided to ERYNInference")
        
        # Store all likelihood kwargs
        self.like_kwargs = kwargs
        
        # We'll use LogLikeERYN which handles the coords->sources conversion
        self.log_like_func = log_like_func
        
        # Get ERYN-specific configuration from nested 'eryn' section
        eryn_config = config['inference']

        self.nwalkers = eryn_config.get('nwalkers', 50)
        self.nsteps = eryn_config.get('nsteps', 10000)
        self.nleaves_max = eryn_config.get('max_nodes', 20)
        self.nleaves_min = eryn_config.get('min_nodes', 0)
        self.ntemps = eryn_config.get('ntemps', 1)  # 1 means no parallel tempering
        self.burnin = eryn_config.get('burnin', 0)
        self.thin = eryn_config.get('thin', 1)
        # Get delta range (can be overridden by sources file)
        self.delta_range = eryn_config.get('delta_range', [-5.0, 5.0])

        self.branches = self._branches()
        self.nbranches = len(self.branches)

        self.nleaves_max, self.nleaves_min = self._nleaves_limits()
    
        self.ndims = self._ndims()

        #print configuration summary
        logger.info("ERYN Inference Configuration:")
        logger.info(f"  Number of walkers: {self.nwalkers}")
        logger.info(f"  Number of steps: {self.nsteps}")
        logger.info(f"  Number of branches: {self.nbranches}")
        logger.info(f"  Number of temperatures: {self.ntemps}")
        logger.info(f"  Branch names: {self.branches}")
        logger.info(f"  Burn-in steps: {self.burnin}")
    
    def _branches(self):
        """
        Get name of branches.
        Returns:
        - list: List of branch names.
        """
        return list(self.sources.keys()) + ['egp']

    def _ndims(self):
        """
        Get number of dimensions for each branch.
        
        Returns:
        - dict: Dictionary with number of dimensions per branch.
        """
        ndims = {}
        
        # Count scalar parameters
        for source_name, param_config in self.sources.items():
            ndims[source_name] = 0
            for param in param_config:
                if param['bounds']:
                    ndims[source_name] += 1
        #add EGP branch
        # Each EGP node has 1 delta parameter
        ndims['egp'] = 1    

        
        return ndims
    
    def _nleaves_limits(self):
        """
        Get nleaves limits for each branch.
        
        Returns:
        - dict: Dictionary with nleaves limits per branch.
        """
        nleaves_max = {}
        nleaves_min = {}

        for source_name, param_config in self.sources.items():
            nleaves_max[source_name] = 1
            nleaves_min[source_name] = 1

        # EGP branch
        nleaves_max['egp'] = self.nleaves_max
        nleaves_min['egp'] = self.nleaves_min

        return nleaves_max, nleaves_min
    
    def _setup_priors(self):
        """
        Create priors for each ERYN branch using ProbDistContainer.

        Returns:
            priors (dict): {branch_name: ProbDistContainer}
        """
        priors = {}

        # Non-EGP branches: parameters from self.sources
        for source_name, param_config in self.sources.items():
            priors_in = {}
            dim_index = 0
            for param in param_config:
                pname = param['name']
                bounds = param.get('bounds')
                if bounds is None:
                    # Parameter has no bounds => Skip (dim is not included)
                    logger.debug(f"Skipping param {pname} in {source_name} (no bounds).")
                    continue

                low, high = float(bounds[0]), float(bounds[1])
                priors_in[dim_index] = uniform_dist(low, high)
                dim_index += 1

            if len(priors_in) == 0:
                # Valid empty branch container
                priors[source_name] = ProbDistContainer({})
            else:
                priors[source_name] = ProbDistContainer(priors_in)

        # EGP branch: 1D delta prior
        low, high = float(self.delta_range[0]), float(self.delta_range[1])
        priors['egp'] = ProbDistContainer({0: uniform_dist(low, high)})

        logger.info("ERYN priors initialized for branches: " + ", ".join(priors.keys()))
        return priors 
                     
    def setup_sampler(self):
        """
        Set up the ERYN ensemble sampler with priors and moves.
        
        Returns:
        - EnsembleSampler: Configured ERYN sampler.
        """
        # Setup priors
        self.priors = self._setup_priors()
        
        # Setup moves (needs priors for RJ generation)
        moves = self._setup_moves()

        #get list of kwargs to pass to likelihood function as args
        args = [self.like_kwargs[key] for key in self.like_kwargs]

        # Create sampler
        self.sampler = EnsembleSampler(
            nwalkers=self.nwalkers,
            ndims=self.ndims,
            log_like_fn=self.log_like_func,
            priors=self.priors,
            args = args,
            tempering_kwargs=dict(ntemps=self.ntemps),
            nbranches=self.nbranches,
            branch_names=self.branches,
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            provide_groups=True,
            moves=moves,
            rj_moves = True
        )

        return self.sampler
    
    def _setup_moves(self):
        """
        Build ERYN proposal moves (Gaussian + RJ) for all branches.
        Returns a flat list of (move, weight) tuples suitable for EnsembleSampler.
        """
        moves = []
        
        # Build covariance matrices for ALL branches at once
        cov_all = {}
        for branch in self._branches():
            ndim = self.ndims[branch]
            if ndim > 0:
                scale = 0.1
                cov_all[branch] = (scale ** 2) * np.eye(ndim)
        
        # Add Gaussian moves for each branch with non-zero dimensions
        for branch in self._branches():
            ndim = self.ndims[branch]
            if ndim > 0:
                moves.append((
                    GaussianMove(
                        branch=branch,
                        cov_all=cov_all,  # Pass complete cov_all dict
                        scale=1.0,
                        adapt_scale=True
                    ),
                    1.0  # weight
                ))
        
        # Add RJ moves for each branch
        for branch in self._branches():
            moves.append((
                DistributionGenerateRJ(
                    generate_dist=self.priors,  
                    nleaves_min=self.nleaves_min,  
                    nleaves_max=self.nleaves_max,  
                    add_prob=0.3,
                    remove_prob=0.3,
                ),
                1.0  # weight
            ))

        return moves


    def _initialize_ensemble(self):
        """
        Initialize walker positions for the ensemble.
        
        Returns:
        - inds: dict of branch -> bool array [ntemps, nwalkers, nleaves_max]
        - coords: dict of branch -> array [ntemps, nwalkers, nleaves_max, ndim]
        """
        inds = {}
        coords = {}

        for branch in self._branches():
            ndim = self.ndims[branch]
            nleaves_max = self.nleaves_max[branch]

            # shape: [ntemps, nwalkers, nleaves_max]
            inds[branch] = np.zeros((self.ntemps, self.nwalkers, nleaves_max), dtype=bool)

            # shape: [ntemps, nwalkers, nleaves_max, ndim]
            coords[branch] = np.zeros((self.ntemps, self.nwalkers, nleaves_max, ndim))

            # Initialize positions for all walkers and leaves
            for temp_idx in range(self.ntemps):
                for walker_idx in range(self.nwalkers):
                    # Only initialize active leaves (here assume starting with 0 leaves)
                    # But rvs expects size = number of leaves to sample
                    nleaves_start = 1  # or self.nleaves_min[branch], adjust as needed
                    samples = self.sampler.priors[branch].rvs(size=nleaves_start)

                    # Assign to first leaves
                    coords[branch][temp_idx, walker_idx, :nleaves_start, :] = samples
                    # Mark leaves as used
                    inds[branch][temp_idx, walker_idx, :nleaves_start] = True

        return inds, coords

    def run(self):
        """
        Run ERYN sampling.
        
        Returns:
        - dict: Results dictionary containing chains and diagnostics.
        """
        if self.sampler is None:
            raise ValueError("Sampler not set up. Call setup_sampler() first.")
        
        # Initialize ensemble
        inds, coords = self._initialize_ensemble()
        

        logger.info("Computing initial log-prior and log-likelihood...")
        log_p = self.sampler.compute_log_prior(coords, inds = inds)
        log_l_raw = self.sampler.compute_log_like(coords, inds=inds, logp=log_p)

        # Extract the array from the tuple if necessary
        if isinstance(log_l_raw, tuple):
            log_l = log_l_raw[0]  # Get the first element (the actual log-likelihood array)
            logger.info(f"Extracted log_l from tuple, shape: {log_l.shape}")
        else:
            log_l = log_l_raw

        state = State(coords, log_like=log_l, log_prior=log_p, inds = inds)
        # Run sampler
        logger.info(f"Running ERYN sampler for {self.nsteps} steps...")
        
        self.sampler.run_mcmc(state, nsteps=self.nsteps, burn = self.burnin, progress=True, thin_by=self.thin)
        
        logger.info("ERYN sampling complete")
        
        # Extract results
        results = self._extract_results()
        
        return results
    
    def _extract_results(self):
        """
        Extract and process results from ERYN sampler.
        
        Returns:
        - dict: Processed results including chains and diagnostics.
        """
        # Get chains - this returns a dict with branch names as keys
        chain = self.sampler.get_chain()
        
        # ERYN has get_log_posterior() which returns: beta*log_like + log_prior
        # We want the untempered posterior (temper=False, beta=1)
        try:
            # Get untempered log-posterior (log_like + log_prior)
            log_prob = self.sampler.get_log_posterior(temper=False)
            logger.info(f"Log-posterior shape: {log_prob.shape}")
            
            # If using parallel tempering, extract the cold chain (temp_index=0)
            if log_prob.ndim == 3:
                log_prob = log_prob[:, 0, :]  # Take cold chain: (nsteps, nwalkers)
                logger.info(f"Extracted cold chain log-posterior: shape {log_prob.shape}")
                
        except AttributeError as e:
            logger.warning(f"Could not extract log-posterior from sampler: {e}")
            logger.warning("Creating placeholder zeros for log-posterior.")
            # Will create dummy array after we know dimensions
            log_prob = None
        except Exception as e:
            logger.error(f"Unexpected error extracting log-posterior: {e}")
            log_prob = None
        
        # Extract EGP chain
        egp_chain = chain['egp']  # Shape could be (nsteps, nwalkers, ...) or (nsteps, ntemps, nwalkers, ...)
        logger.info(f"EGP chain shape: {egp_chain.shape}")
        
        # Extract scalar chains for each non-EGP branch
        scalar_branches = [b for b in self.branches if b != 'egp']
        scalar_chains = {branch: chain[branch] for branch in scalar_branches}
        
        # Get dimensions from first chain and handle temperature dimension
        first_chain = list(chain.values())[0]
        logger.info(f"First chain shape: {first_chain.shape}")
        
        # Determine if we have temperature dimension
        if first_chain.ndim >= 4:
            # Shape: (nsteps, ntemps, nwalkers, nleaves, ndim) - extract cold chain
            has_temps = True
            nsteps = first_chain.shape[0]
            ntemps = first_chain.shape[1]
            nwalkers = first_chain.shape[2]
            logger.info(f"Detected parallel tempering: nsteps={nsteps}, ntemps={ntemps}, nwalkers={nwalkers}")
        else:
            # Shape: (nsteps, nwalkers, nleaves, ndim)
            has_temps = False
            nsteps = first_chain.shape[0]
            nwalkers = first_chain.shape[1]
            logger.info(f"No parallel tempering: nsteps={nsteps}, nwalkers={nwalkers}")
        
        # Extract cold chain if using parallel tempering
        if has_temps:
            egp_chain = egp_chain[:, 0, :, :, :]  # Extract cold chain
            scalar_chains = {branch: scalar_chains[branch][:, 0, :, :, :] for branch in scalar_branches}
            logger.info(f"Extracted cold chain - EGP shape: {egp_chain.shape}")
        
        # Create log_prob if it wasn't extracted successfully
        if log_prob is None:
            logger.warning(f"Creating placeholder log_prob with shape ({nsteps}, {nwalkers})")
            log_prob = np.zeros((nsteps, nwalkers))
        
        # Combine all scalar parameters into a single array
        scalar_list = []
        scalar_names = []
        for branch in scalar_branches:
            branch_chain = scalar_chains[branch]  # Shape: (nsteps, nwalkers, nleaves=1, ndim)
            logger.info(f"Branch '{branch}' chain shape: {branch_chain.shape}")
            
            # Extract from the single leaf
            branch_flat = branch_chain[:, :, 0, :]  # Shape: (nsteps, nwalkers, ndim)
            logger.info(f"Branch '{branch}' after leaf extraction: {branch_flat.shape}")
            
            # Get parameter names for this branch
            for param in self.sources[branch]:
                if param.get('bounds') is not None:
                    scalar_names.append(f"{branch}:{param['name']}")
            
            # Flatten walker dimension - handle both 2D and 3D cases
            if branch_flat.ndim == 3:
                # Shape: (nsteps, nwalkers, ndim)
                ndim = branch_flat.shape[2]
                branch_flat = branch_flat.reshape(nsteps * nwalkers, ndim)
            elif branch_flat.ndim == 2:
                # Shape: (nsteps, nwalkers) - single parameter, no extra dimension
                branch_flat = branch_flat.reshape(nsteps * nwalkers, 1)
            else:
                logger.error(f"Unexpected branch_flat shape: {branch_flat.shape}")
                continue
                
            logger.info(f"Branch '{branch}' after reshape: {branch_flat.shape}")
            scalar_list.append(branch_flat)
        
        # Concatenate all scalar parameters
        if scalar_list:
            scalar_chain_flat = np.column_stack(scalar_list)
        else:
            scalar_chain_flat = np.zeros((nsteps * nwalkers, 0))
        
        # Get number of leaves at each step
        nleaves_dict = self.sampler.get_nleaves()
        nleaves = nleaves_dict['egp']  # Shape could be (nsteps, nwalkers) or (nsteps, ntemps, nwalkers)
        logger.info(f"nleaves shape: {nleaves.shape}")
        
        # Handle different nleaves shapes
        if nleaves.ndim == 3:
            # Shape: (nsteps, ntemps, nwalkers) - extract cold chain
            nleaves = nleaves[:, 0, :]  # Shape: (nsteps, nwalkers)
            logger.info(f"Extracted cold chain nleaves: {nleaves.shape}")
        
        # Flatten chains (combine walkers)
        egp_chain_flat = egp_chain.reshape(nsteps * nwalkers, *egp_chain.shape[2:])
        nleaves_flat = nleaves.reshape(nsteps * nwalkers)
        log_prob_flat = log_prob.reshape(nsteps * nwalkers)
        
        # Calculate statistics
        mean_nodes = np.mean(nleaves_flat)
        mode_nodes = int(np.bincount(nleaves_flat.astype(int)).argmax())
        
        # Get acceptance fraction safely
        try:
            acceptance = self.sampler.acceptance_fraction
            logger.info(f"Extracted acceptance_fraction")
        except AttributeError:
            logger.warning("Could not extract acceptance fraction. Computing manually.")
            acceptance = self._compute_acceptance_rate(chain)
        
        results = {
            'egp_chain': egp_chain_flat,
            'scalar_chain': scalar_chain_flat,
            'scalar_names': scalar_names,
            'nleaves': nleaves_flat,
            'log_like': log_prob_flat,  # Actually log_posterior, but name kept for compatibility
            'mean_nodes': mean_nodes,
            'mode_nodes': mode_nodes,
            'acceptance_fraction': acceptance,
            'branches': self.branches,
            'sources': self.sources
        }
        
        logger.info(f"Extracted results:")
        logger.info(f"  Mean number of nodes: {mean_nodes:.2f}")
        logger.info(f"  Mode number of nodes: {mode_nodes}")
        if isinstance(acceptance, np.ndarray):
            logger.info(f"  Mean acceptance fraction: {np.mean(acceptance):.3f}")
        elif acceptance is not None:
            logger.info(f"  Acceptance fraction: {acceptance:.3f}")
        logger.info(f"  Scalar parameters: {scalar_names}")
        
        return results
    
    def _compute_acceptance_rate(self, chain):
        """
        Compute acceptance rate by detecting changes in chain.
        
        Parameters:
        -----------
        chain : dict
            Dictionary of chains from sampler
        
        Returns:
        --------
        acceptance : float
            Overall acceptance rate
        """
        # Use the first branch to detect changes
        first_branch = list(chain.keys())[0]
        branch_chain = chain[first_branch]
        
        nsteps, nwalkers = branch_chain.shape[:2]
        
        # Count steps where state changed
        changes = 0
        total = 0
        
        for step in range(1, nsteps):
            for walker in range(nwalkers):
                current = branch_chain[step - 1, walker]
                next_state = branch_chain[step, walker]
                
                # Check if state changed
                if not np.allclose(current, next_state, rtol=1e-12, atol=1e-12):
                    changes += 1
                total += 1
        
        return changes / total if total > 0 else 0.0



