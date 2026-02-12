import torch
import math

class LinearGaussianProcess():
    """
    Gaussian process with kernel set to a linear kernel in some feature space. The free prior parameters (nu, lambda) can be fit with marginal likelihood
    maximization and affect the kernel through k_prior(x,z) = lambda^2 * phi^T(x) phi(z) and mu_prior(x) = nu, where phi(x) are the features on which the 
    linear kernel acts. This implementation has the following runtimes:

    - initialization: Theta(d^2)
    - add k observations: Theta(d^2 * k)
    - marginal likelihood maximization: Theta(d^2)
    - compute posterior mean at k points: Theta(d^2 + d * k)
    - compute posterior variance at k points: Theta(d^2 * k)
    - compute posterior covariance at k points: Theta(d^2 * k + d * k^2)
    """
    def __init__(self, 
                 n_features:int, 
                 nar:float, 
                 nu:float=0.0, 
                 lambd:float=1.0,
                 device='cpu',
                 dtype=None):
        """
        Initialize the LinearGaussianProcess with specified parameters.

        Args:
            n_features (int): The number of features in the input data.
            nar (float): The noise-to-amplitude ratio, where the amplitude is given by lambd. This parameter affects the observation covariance matrix.
            nu (float, optional): The prior mean parameter. Defaults to 0.0.
            lambd (float, optional): The amplitude parameter for the kernel. Defaults to 1.0.
            device (str, optional): The device on which to perform computations. Defaults to 'cpu'.
            dtype (optional): The data type of the tensors. Defaults to None, which retrieves the default torch dataype.
        """
        if dtype is None:
            dtype = torch.get_default_dtype() 
        self._dtype = dtype # double precision may be necessary for stable inverses (Shermann-Morrison-Woodbury matrix identity is known for numerical stability issues)
        self._device = device
        self._n_features = n_features
        self._nar = nar
        self._n_observations = 0
        # prior parameters that are updated with marginal likelihood maximization
        self._nu = nu
        self._lambd = lambd
        # inverse feature + noise autocorrelation matrix (Phi Phi^T + _nar^2 * I)^{-1}
        self._inv_Psi = torch.eye(n_features, n_features, device=device, dtype=dtype) / (nar**2)
        # feature autocorrelation matrix Phi Phi^T
        self._Phi_times_Phi_transpose = torch.zeros((n_features, n_features), device=device, dtype=dtype)
        # cached multiplication of Phi with vectors that scale in the number of observations points
        self._Phi_times_y_O = torch.zeros((n_features), device=device, dtype=dtype)
        self._Phi_times_ones = torch.zeros((n_features), device=device, dtype=dtype)
        # additional necessary_statistics for closed-form marginal likelihood maximization
        self._y_O_inner_y_O = 0
        self._y_O_inner_ones = 0
        # record memory consumption (inv_Psi, Phi_times_Phi_transpose, Phi_times_y_0, Phi_times_ones)
        self._memory = (2 * n_features**2 + 2 * n_features) * self._inv_Psi.element_size()

    @property
    def n_features(self) -> int:
        return self._n_features
    @property
    def nar(self) -> float:
        return self._nar
    @property
    def n_observations(self) -> int:
        return self._n_observations
    @property
    def memory(self) -> int:
        """# of bytes used by the model"""
        return self._memory
    @property
    def device(self) -> torch.device:
        return self._device

    def marginal_likelihood_maximization(self, min_obs:int=20, print_result:bool=True) -> None:
        """
        Updates the prior parameters using marginal likelihood optimization, all in Theta(d^2)

        Args:
            min_obs (int, optional): The minimum number of observations required to perform marginal likelihood maximization. Defaults to 20.
            print_result (bool, optional): Whether to print the MLM optimization result. Defaults to True.
        """
        if self.n_observations < min_obs:
            return
        # the Sherman–Morrison–Woodbury formula states that (A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
        # which specializes to (nar^2 * I + Phi^T Phi)^{-1} = I / nar^2 - Phi^T (nar^2 * I + Phi Phi^T)^{-1} Phi / nar^2 = () / nar^2
        y_O_times_inv_Sigma_times_y_O   = (self._y_O_inner_y_O  - self._Phi_times_y_O.t()  @ self._inv_Psi @ self._Phi_times_y_O)  / self._nar**2
        ones_times_inv_Sigma_times_ones = (self._n_observations - self._Phi_times_ones.t() @ self._inv_Psi @ self._Phi_times_ones) / self._nar**2
        y_O_times_inv_Sigma_times_ones  = (self._y_O_inner_ones - self._Phi_times_y_O.t()  @ self._inv_Psi @ self._Phi_times_ones) / self._nar**2

        self._nu = y_O_times_inv_Sigma_times_ones / ones_times_inv_Sigma_times_ones
        self._lambd = max(0, (y_O_times_inv_Sigma_times_y_O + self._nu**2 * ones_times_inv_Sigma_times_ones - 2 * self._nu * y_O_times_inv_Sigma_times_ones) / self._n_observations)**.5
        if print_result:
            print(f"MLM: prior-mean={self._nu}, prior-std={self._lambd}")
        #assert math.isfinite(self._nu)
        #assert math.isfinite(self._lambd)

    def _add_observations(self, 
                          batched_features:torch.Tensor, 
                          batched_obs:torch.Tensor):
        """
        Updates the cache and statistics with the new features and observations, all in Theta(d^2 * k + d * k^2 + k^3)

        Args:
            batched_features (torch.Tensor): The features associated with the new observations.
            batched_obs (torch.Tensor): The observations associated with the new features.
        """
        k = batched_obs.nelement()
        delta_phi = batched_features.t() # (d, k)
        # updates cache
        self._Phi_times_y_O += delta_phi @ batched_obs
        self._Phi_times_ones += delta_phi @ torch.ones_like(batched_obs)
        self._y_O_inner_y_O += torch.sum(batched_obs**2)
        self._n_observations += k
        self._y_O_inner_ones += torch.sum(batched_obs)
        # updates Phi Phi^T
        self._Phi_times_Phi_transpose += delta_phi @ delta_phi.t()
        # updates inv_Psi using the Sherman–Morrison–Woodbury formula
        inv_psi_times_delta_phi = self._inv_Psi @ delta_phi # Theta(d^2 * k)
        self._inv_Psi -=  inv_psi_times_delta_phi @ torch.inverse(torch.eye(k, device=self._device, dtype=self._dtype) + delta_phi.t() @ inv_psi_times_delta_phi) @ inv_psi_times_delta_phi.t() # Theta(d^2 * k + d * k^2 + k^3)
        # evens out errors and ensures symmetry
        self._Phi_times_Phi_transpose = (self._Phi_times_Phi_transpose + self._Phi_times_Phi_transpose.t()) / 2
        self._inv_Psi = (self._inv_Psi + self._inv_Psi.t()) / 2
        # asserts correctness of state
        assert self._Phi_times_y_O.isfinite().all()
        assert self._Phi_times_ones.isfinite().all()
        assert self._y_O_inner_y_O.isfinite().all()
        assert self._y_O_inner_ones.isfinite().all()
        assert self._Phi_times_Phi_transpose.isfinite().all()
        assert self._inv_Psi.isfinite().all()
        #print(f"||I - Psi^-1(Phi Phi^T+sigma I)||_F^2): {torch.linalg.norm(torch.eye(n=self._n_features, device=self._device) - self._inv_Psi @ (self._Phi_times_Phi_transpose + self._nar**2 * torch.eye(n=self._n_features, device=self._device)))**2}") # very expensive, but can be used to DEBUG instabilities

    def add_observations(self, 
                         batched_features:torch.Tensor, 
                         batched_obs:torch.Tensor, 
                         perform_marginal_likelihood_maximization:bool=False,
                         min_obs:int=20) -> None:
        """
        Conditions (linear) Gaussian process on additional observations in Theta(d^2 * k)

        Args:
            batched_features (torch.Tensor): Batched features of shape (k, d)
            batched_obs (torch.Tensor): Batched observations of shape (k)
            perform_marginal_likelihood_maximization (bool, optional): Whether to use marginal likelihood maximization to update (prior_mean_offset, prior_mean_scaling, prior_std_scaling). Defaults to False.
            min_obs (int, optional): The least number of observations necessary to perform marginal likelihood maximization. Defaults to 20.
        """
        assert batched_features.ndim == 2 and batched_obs.ndim == 1, "tensors with wrong shapes provided"
        assert batched_features.isfinite().all(), "provided features must be finite"
        assert batched_obs.isfinite().all(), "provided observations must be finite"
        batched_features = batched_features.to(self._device)
        batched_obs = batched_obs.to(self._device)
        batched_features = batched_features.type(self._dtype)
        batched_obs = batched_obs.type(self._dtype)
        splitted_X = torch.split(batched_features, self._n_features, dim=0)
        splitted_Y = torch.split(batched_obs, self._n_features, dim=0)
        for _X, _Y in zip(splitted_X, splitted_Y): # the splitting ensures that k <= d in _add_observations => matrix inverse does not dominate compute cost
            self._add_observations(_X, _Y)
        if perform_marginal_likelihood_maximization:
            self.marginal_likelihood_maximization(min_obs)

    def add_observation(self, 
                        features:torch.Tensor, 
                        obs:torch.Tensor, 
                        perform_marginal_likelihood_maximization:bool=False,
                        min_obs:int=20) -> None:
        """
        Conditions (linear) Gaussian process on one additional observation in Theta(d^2)

        Args:
            features (torch.Tensor): Features of shape (d)
            obs (torch.Tensor): Observation of shape (,)
            perform_marginal_likelihood_maximization (bool, optional): Whether to use marginal likelihood maximization to update (prior_mean_offset, prior_mean_scaling, prior_std_scaling). Defaults to False.
            min_obs (int, optional): The least number of observations necessary to perform marginal likelihood maximization. Defaults to 20.
        """
        assert features.ndim == 1 and obs.ndim == 0, "tensors with wrong shapes provided"
        assert features.isfinite().all(), "provided features must be finite"
        assert obs.isfinite().all(), "provided observation must be finite"
        self.add_observations(features.unsqueeze(0), obs.unsqueeze(0), perform_marginal_likelihood_maximization, min_obs)

    def posterior_mean(self, 
                       features:torch.Tensor) -> torch.Tensor:
        """
        Returns mu(x) + k(x, x_O)^T (k(x_O, x_O) + _nar^2 * I)^{-1} (y_O - mu_O), all in Theta(d * (k+d))

        Args:
            features (torch.Tensor): Features of shape (d), or batched features of shape (k, d)

        Returns:
            out (torch.Tensor): Posterior means of shape (k)
        """
        assert (features.ndim == 2 or features.ndim == 1), "tensor of wrong shapes provided"
        assert features.isfinite().all(), "provided features must be finite"
        features = features.to(self._device)
        features = features.type(self._dtype)
        Phi_times_obs_mean_difference = self._Phi_times_y_O - self._nu * self._Phi_times_ones
        mean_correction = features @ (self._inv_Psi @ Phi_times_obs_mean_difference) 
        return self._nu + mean_correction

    def posterior_cov(self, 
                      features:torch.Tensor) -> torch.Tensor:
        """Returns k(x, z) - k(x, x_O)^T (k(x_O, x_O) + _nar^2 * I)^{-1} k(z, x_O), all in Theta(k * d^2 + k^2 * d)

        Args:
            features (torch.Tensor): Batched features of shape (k, d)

        Returns:
            out (torch.Tensor): Posterior covariance matrix of shape (k, k)
        """
        assert features.ndim == 2, "tensor of wrong shapes provided"
        assert features.isfinite().all(), "provided features must be finite"
        features = features.to(self._device)
        features = features.type(self._dtype)
        covariance = self._lambd**2 * self._nar**2 * torch.einsum('kd,ld->kl', features @ self._inv_Psi, features) 
        return (covariance + covariance.t())/2 # evens out numerical errors to ensure symmetry
    

    def posterior_var(self, 
                      features:torch.Tensor) -> torch.Tensor:
        """Returns k(x, x) - k(x, x_O)^T (k(x_O, x_O) + _nar^2 * I)^{-1} k(x, x_O), all in Theta(k * d^2)

        Args:
            features (torch.Tensor): Features of shape (d), or batched features of shape (k, d)

        Returns:
            out (torch.Tensor): posterior element-wise variance of shape (k)
        """
        assert features.ndim == 2 or features.ndim == 1, "tensor of wrong shapes provided"
        assert features.isfinite().all(), "provided features must be finite"
        features = features.to(self._device)
        features = features.type(self._dtype)
        features = features.reshape(-1, self._n_features) # transforms shape (d) into shape (1, d) but leaves shape (k, d) unchanged
        variance = self._lambd**2 * self._nar**2 * torch.einsum('kd,kd->k', features @ self._inv_Psi, features) 
        assert (variance >= 0).all(), f"posterior variance must be positive but is {variance}"
        return variance