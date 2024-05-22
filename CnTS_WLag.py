import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize, LinearConstraint
import itertools
from copy import deepcopy

PRECISION = 1e-12


######## UTILS ########
def matrix_init(A):
    A_init = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_init.append(np.random.normal(A[i][j],1))
    return np.array(A_init).reshape(A.shape[0],A.shape[1])

def project_on_feasible(allocation, A, b):
    """
    Project allocation on feasible set
    :param allocation: allocation to project
    :param A: matrix of constraints
    :param b: vector of constraints
    """
    simplex = np.ones_like(allocation).reshape(1, -1)
    eye = np.eye(len(allocation))
    if A is not None:
        A = np.concatenate([A, eye, -eye, simplex, -simplex], axis=0)
        b = np.concatenate(
            [
                b,
                np.ones(len(allocation)),
                np.zeros(len(allocation)),
                np.array([1]),
                np.array([-1]),
            ],
            axis=0,
        )
    else:
        A = np.concatenate([eye, -eye, simplex, -simplex], axis=0)
        b = np.concatenate(
            [
                np.ones(len(allocation)),
                np.zeros(len(allocation)),
                np.array([1]),
                np.array([-1]),
            ],
            axis=0,
        )
    constraints = LinearConstraint(A=A, ub=b)
    x0 = np.ones_like(allocation) / len(allocation)
    fun = lambda x, y: np.linalg.norm(x - y) ** 2
    try:
        results = minimize(fun=fun, x0=x0, args=(allocation), constraints=constraints)
        x = results["x"]
    except ValueError:
        pass
    if np.abs(np.sum(x) - 1) > 1e-5:
        raise "Allocation doesnt sum to 1"
    return x


def get_policy(mu, A, b):
    """
    Find optimal policy
    :param mu: Reward vector
    :param A: if None solve standard bandit problem without any constraints on policy
    :param b: if None solve standard bandit problem without any constraints on policy
    :return:
        - optimal policy
        - aux info from optimizer
    """
    simplex = np.ones_like(mu).reshape(1, -1)
    eye = np.eye(len(mu))
    one = np.array([1])
    if A is not None:
        A = np.concatenate([A, -eye, simplex, -simplex], axis=0)
        b = np.concatenate([b, np.zeros(len(mu)), one, -one], axis=0)
    else:
        A = np.concatenate([-eye, simplex, -simplex], axis=0)
        b = np.concatenate([np.zeros(len(mu)), one, -one], axis=0)
    try:
        results = linprog(
            -mu, A_ub=A, b_ub=b, A_eq=None, b_eq=None, method="highs-ds"
        )  # Use simplex method
    except ValueError:
        pass
    #print(results)
    #if not results["success"]:
    #    raise "LP Solver failed"
    # Get active constraints
    aux = {"A": A, "b": b, "slack": results["slack"]}
    return results,aux


def arreqclose_in_list(myarr, list_arrays):
    """
    Test if np array is in list of np arrays
    """
    return next(
        (
            True
            for elem in list_arrays
            if elem.size == myarr.size and np.allclose(elem, myarr)
        ),
        False,
    )


def enumerate_all_policies(A, b):
    """
    Enumerate all policies in the polytope Ax <= b
    """
    # Compute all possible bases
    n_constraints = A.shape[0]
    n_arms = A.shape[1]
    bases = list(itertools.combinations(range(n_constraints), n_arms))  #Takes all possible sub-matrices
    policies = []
    for base in bases:
        base = np.array(base)
        B = A[base]
        # Check that the base is not degenerate
        if np.linalg.matrix_rank(B) == A.shape[1]:
            policy = np.linalg.solve(B, b[base])
            # Verify that policy is in the polytope
            if np.all(A.dot(policy) <= b + 1e-5) and not arreqclose_in_list(
                policy, policies
            ):
                policies.append(policy)
    return policies


def compute_neighbors(vertex, A, b, slack):
    """
    Compute all neighbors of vertex in the polytope Ax <= b
    :param vertex: vertex of the polytope
    :param A: matrix of constraints
    :param b: vector of constraints
    :param slack: vector of slack variables
    """
    active = slack == 0
    not_active = slack != 0
    #print(A)
    n_constraints = np.arange(A.shape[0])
    #print(active)
    #print(n_constraints)
    active_constaints = n_constraints[active].tolist()
    inactive_constraints = n_constraints[not_active].tolist()
    neighbors = []

    # Compute all possible bases at the vertex
    bases = list(itertools.combinations(active_constaints, len(vertex)))
    # For each possible base swap one element with an inactive constraint to get a neighbor
    for base in bases:
        for constraint in inactive_constraints:
            # Swap constraint into each position of the base
            for i in range(len(base)):
                new_base = np.array(deepcopy(base))
                new_base[i] = constraint
                B = A[new_base]
                # Check that the base is not degenerate
                if np.linalg.matrix_rank(B) == len(vertex):
                    possible_neighbor = np.linalg.solve(B, b[new_base])
                    # Verify that neighbor is in the polytope
                    if np.all(
                        A.dot(possible_neighbor) <= b + 1e-5
                    ) and not arreqclose_in_list(possible_neighbor, neighbors):
                        neighbors.append(possible_neighbor)
    return neighbors


def binary_search(mu, interval, threshold, kl):
    """
    Find maximizer of KL(mu, x) in interval satysfiyng threshold using binary search
    :param mu: reward of arm
    :param interval: interval to search in
    :param threshold: threshold to satisfy (f(t) = log t)
    :param kl: KL divergence function

    """
    p = 0
    q = len(interval)
    done = False
    while not done:
        i = int((p + q) / 2)
        x = interval[i]
        loss = kl(mu, x)
        if loss < threshold:
            p = i
        else:
            q = i
        if p + 1 >= q:
            done = True

    return x, loss


def get_confidence_interval(mu, pulls, f_t, upper=6, lower=-1, kl=None):
    """
    Compute confidence interval for each arm
    :param mu: reward vector
    :param pulls: number of pulls for each arm
    :param f_t: threshold function f(t) = log t
    :param upper: upper bound for search
    :param lower: lower bound for search
    :param kl: KL divergence function
    :return:
        - lower bound for each arm
        - upper bound for each arm
    """
    if kl is None:
        kl = lambda m1, m2: ((m1 - m2) ** 2) / (2)
    ub = [
        binary_search(m, np.linspace(m, upper, 5000), threshold=f_t / n, kl=kl)[0]
        for m, n in zip(mu, pulls)
    ]
    lb = [
        binary_search(m, np.linspace(m, lower, 5000), threshold=f_t / n, kl=kl)[0]
        for m, n in zip(mu, pulls)
    ]

    return lb, ub


def gaussian_projection(w, mu, pi1, pi2, sigma=1):
    """
    perform close-form projection onto the hyperplane lambda^T(pi1 - pi2) = 0 assuming Gaussian distribution
    :param w: weight vector
    :param mu: reward vector
    :param pi1: optimal policy
    :param pi2: suboptimal neighbor policy
    :param sigma: standard deviation of Gaussian distribution

    return:
        - lambda: projection
        - value of the projection

    """
    v = pi1 - pi2
    normalizer = ((v**2) / (w + PRECISION)).sum()
    lagrange = mu.dot(v) / (normalizer+PRECISION)
    lam = mu - lagrange * v / (w + PRECISION)
    var = sigma**2
    value = (w * ((mu - lam) ** 2)).sum() / (2 * var)
    return lam, value


def bernoulli_projection(w, mu, pi1, pi2, sigma=1):
    """
    Projection onto the hyperplane lambda^T(pi1 - pi2) = 0 assuming Bernoulli distribution using scipy minimize
    """
    mu = np.clip(mu, 1e-3, 1 - 1e-3)
    bounds = [(1e-3, 1 - 1e-3) for _ in range(len(mu))]
    v = pi1 - pi2
    constraint = LinearConstraint(v.reshape(1, -1), 0, 0)

    def objective(lam):
        kl_bernoulli = mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))
        return (w * kl_bernoulli).sum()

    x0 = gaussian_projection(w, mu, pi1, pi2, sigma)[0]
    x0 = np.clip(x0, 1e-3, 1 - 1e-3)
    res = minimize(objective, x0, constraints=constraint, bounds=bounds)
    lam = res.x
    value = objective(lam)
    return lam, value


def best_response(w, mu, pi, neighbors, sigma=1, dist_type="Gaussian"):
    """
    Compute best response instance w.r.t. w by projecting onto neighbors
    :param w: weight vector
    :param mu: reward vector
    :param pi: optimal policy
    :param neighbors: list of neighbors
    :param sigma: standard deviation of Gaussian distribution
    :param dist_type: distribution type to use for projection

    return:
        - value of best response
        - best response instance
    """
    if dist_type == "Gaussian":
        projections = [
            gaussian_projection(w, mu, pi, neighbor, sigma) for neighbor in neighbors
        ]
    elif dist_type == "Bernoulli":
        projections = [
            bernoulli_projection(w, mu, pi, neighbor, sigma) for neighbor in neighbors
        ]
    else:
        raise NotImplementedError
    values = [p[1] for p in projections]
    instances = [p[0] for p in projections]
    return np.min(values), instances[np.argmin(values)]


def solve_game(
    mu,
    vertex,
    neighbors,
    sigma=1,
    dist_type="Gaussian",
    allocation_A=None,
    allocation_b=None,
    tol=None,
    x0=None,
):
    """
    Solve the game instance w.r.t. reward vector mu. Used for track-n-stop algorithms
    :param mu: reward vector
    :param vertex: vertex of the game
    :param neighbors: list of neighbors
    :param sigma: standard deviation of Gaussian distribution
    :param dist_type: distribution type to use for projection
    :param allocation_A: allocation constraint. If None allocations lies in simplex.
    :param tol: Default None for speed in TnS.
    :param x0: initial point
    """

    def game_objective(w):
        return -best_response(w, mu, vertex, neighbors, sigma, dist_type)[0]

    tol_sweep = [1e-16, 1e-12, 1e-6, 1e-4]  # Avoid tolerance issues in scipy
    if tol is not None and tol > tol_sweep[0]:
        tol_sweep = [tol] + tol_sweep
    else:
        tol_sweep = [None] + tol_sweep  # Auto tune tol via None
    if allocation_A is None:
        # Solve optimization problem over simplex
        simplex = np.ones_like(mu).reshape(1, -1)
        constraint = LinearConstraint(A=simplex, lb=1, ub=1)
        bounds = [(0, 1) for _ in range(len(mu))]
        count = 0
        done = False
        if x0 is None:
            while count < len(tol_sweep) and not done:
                x0 = np.random.uniform(0.3, 0.6, size=len(mu))
                x0 = x0 / x0.sum()
                tol = tol_sweep[count]
                res = minimize(
                    game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
                )
                done = res["success"]
                count += 1
        else:
            res = minimize(
                game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
            )
    else:
        # Solve optimization problem over allocation constraint
        constraint = LinearConstraint(A=allocation_A, ub=allocation_b)
        bounds = [(0, 1) for _ in range(len(mu))]
        count = 0
        done = False
        if x0 is None:
            while count < len(tol_sweep) and not done:
                tol = tol_sweep[count]
                x0 = np.random.uniform(0.3, 0.6, size=len(mu))
                x0 = project_on_feasible(x0, allocation_A, allocation_b)
                res = minimize(
                    game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
                )
                done = res["success"]
                count += 1
        else:
            res = minimize(
                game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
            )
    if res["success"] == False:
        #raise ValueError("Optimization failed")
        pass
    return res.x, -res.fun


####### ALGORITHMS #######


class Explorer:
    """
    Abstract class for an explorer
    """

    def __init__(
        self,
        n_arms,
        A_init,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=True,
        dist_type="Gaussian",
        seed=None,
        d_tracking=False,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts). Default: 1
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        """
        self.n_arms = n_arms
        self.A_init = A_init
        self.b = b
        self.delta = delta
        self.ini_phase = ini_phase
        self.sigma = sigma
        self.restricted_exploration = restricted_exploration
        self.dist_type = dist_type
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.d_tracking = d_tracking
        self.cumulative_weights = np.zeros(n_arms)
        self.D = 1
        self.alpha = 1

        self.t = 0
        self.neighbors = {}
        self.means = np.zeros(n_arms)
        self.constraints = A_init
        self.n_pulls = np.zeros(n_arms)
        self.res = []
        self.au = []
        self.gram_mat = np.eye(n_arms)

        if dist_type == "Gaussian":
            # Set KL divergence and lower/upper bounds for binary search
            self.kl = lambda x, y: 1 / (2 * (sigma**2)) * ((x - y) ** 2)
            self.lower = -1
            self.upper = 10
        elif dist_type == "Bernoulli":
            # Set KL divergence for Bernoulli distribution and lower/upper bounds for binary search
            self.kl = lambda x, y: x * np.log(x / y) + (1 - x) * np.log(
                (1 - x) / (1 - y)
            )
            self.lower = 0 + 1e-4
            self.upper = 1 - 1e-4
            self.ini_phase = (
                10  # Take longer initial phase for Bernoulli to avoid all 0  or all 1
            )
        else:
            raise NotImplementedError

        if restricted_exploration:
            # Compute allocation constraint
            test = np.ones_like(self.means)
            _,aux = get_policy(test, A=self.constraints, b=self.b)
            aux = {"A": self.constraints, "b": aux["b"], "slack": aux["slack"]} 
            self.allocation_A = aux["A"]
            self.allocation_b = aux["b"]
        else:
            self.allocation_A = None
            self.allocation_b = None

    def tracking(self, allocation):
        """
        Output arm based on either d-tracking or cumulative tracking
        """
        if self.d_tracking:
            return np.argmin(self.n_pulls - self.t * allocation)
        else:
            eps = 1 / (2 * np.sqrt(self.t + self.n_arms**2))
            eps_allocation = allocation + eps
            eps_allocation = eps_allocation / eps_allocation.sum()
            self.cumulative_weights += eps_allocation
            return np.argmin(self.n_pulls - self.cumulative_weights)

    def act():
        """
        Choose an arm to play
        """
        raise NotImplementedError

    def stopping_criterion(self, vertex, arm, f_t, w_t_norm):
        """
        Check stopping criterion. Stopping based on the generalized log-likelihood ratio test
        """

        hash_tuple = tuple(vertex.tolist())
        game_value, _ = best_response(
            w=self.empirical_allocation(),
            mu=self.means,
            pi=vertex,
            neighbors=self.neighbors[hash_tuple],
            sigma=self.sigma,
            dist_type=self.dist_type,
        )
        #print(game_value)
        #print(self.gram_mat)
        beta = np.log(2*(1 + np.log(self.t)) / self.delta)
        #print(beta)
        opt_const = self.constraints - f_t*np.sqrt(w_t_norm)
        return self.t * game_value + np.absolute((self.b-opt_const.dot(vertex))).sum() > beta + self.gram_mat.shape[0]*f_t*np.sqrt(w_t_norm)


    def empirical_allocation(self):
        """
        Compute empirical allocation
        """
        return self.n_pulls / self.t

    def update(self, arm, reward, cost):     # Update empirical estimates
        """
        Update the explorer with the reward obtained from playing the arm
        :param arm: arm played
        :param reward: reward obtained
        """
        self.noise = 1
        self.t += 1
        #print(self.t)
        self.n_pulls[arm] += 1
        self.means[arm] = self.means[arm] + (1 / self.n_pulls[arm]) * (
            reward - self.means[arm]
        )
        self.constraints.T[arm] = ((self.n_pulls[arm]-1)*self.constraints.T[arm] + cost)/self.n_pulls[arm] 
        self.gram_mat += np.outer(np.eye(A.shape[1])[arm],np.eye(A.shape[1])[arm])
        #print(self.constraints)
        
        if self.dist_type == "Bernoulli":
            self.means = np.clip(self.means, self.lower, self.upper)

class TnS(Explorer):
    """
    Track-n-stop style of algorithm for bandits with linear constraints
    """

    def __init__(
        self,
        n_arms,
        A_init,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=True,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts)
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        :param d_tracking: D-tracking or C-tracking
        """
        super().__init__(
            n_arms,
            A_init,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )

    def act(self):
        """
        Choose an arm to play
        """
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None
        
        w = 2*len(self.means)*np.log(1+1/(len(self.means)+self.t))
        #f_t = 1 + np.sqrt((1/2) * np.log(4*np.sqrt(np.linalg.det(self.gram_mat))/self.delta))
        f_t = 1 + np.sqrt((1/2) * np.log(4/self.delta)
                                        + (len(self.means)/4) * np.log(1 + self.t/len(self.means)))
        # Compute optimal policy w.r.t. current empirical means
        #print(self.constraints)
        #print(len(self.res))
        results,aux= get_policy(mu=self.means, A = self.constraints, b=self.b)        
        optimal_policy = results["x"]


        if results["success"] == True:   

            w_t_norm = np.matmul(np.matmul(optimal_policy, np.linalg.inv(self.gram_mat)),optimal_policy)
            # Check if policy already visited. If yes retrieve neighbors otherwise compute neighbors
            hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
            if hash_tuple in self.neighbors:
                neighbors = self.neighbors[hash_tuple]
            else:
                neighbors = compute_neighbors(
                    optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
                )
                self.neighbors[hash_tuple] = neighbors

            # Solve game to get allocation
            allocation, game_value = solve_game(
                mu=self.means,
                vertex=optimal_policy,
                neighbors=neighbors,
                dist_type=self.dist_type,
                sigma=self.sigma,
                allocation_A=self.constraints-f_t*np.sqrt(w_t_norm),
                allocation_b=self.b,
            )
            if allocation.any() != None:
                # Check if forced exploration is needed else D-tracking
                not_saturated = self.n_pulls < (np.sqrt(self.t) - self.n_arms / 2)
                if not_saturated.any() and self.d_tracking:
                    # Play smallest N below sqrt(t) - n_arms/2
                    arm = np.argmin(self.n_pulls)

                else:
                    # Play arm according to tracking rule
                    arm = self.tracking(allocation)

                
                # Check stopping criterion
                stop = self.stopping_criterion(optimal_policy,arm,f_t,w_t_norm)

                misc = {
                    "game_value": game_value,
                    "allocation": allocation,
                    "optimal_policy": optimal_policy
                }

                return arm, stop, optimal_policy, misc


class UniformExplorer(Explorer):
    """
    Uniform explorer for bandits with linear constraints. If restricted during exploration, the uniform policy is projected onto the feasible set.
    """

    def __init__(
        self,
        n_arms,
        A_init,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        allocation=None,
    ):
        super().__init__(
            n_arms,
            A_init,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
        )

        if allocation is None:
            self.allocation = np.ones(n_arms) / n_arms
        else:
            self.allocation = allocation

        #if self.restricted_exploration:
        #    self.allocation = project_on_feasible(self.allocation, self.constraints, self.b)

    def act(self):
        """
        Choose an arm to play
        """

        # Play each arm once
        if self.t < self.n_arms * self.ini_phase:
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Get optimal policy
        results,aux= get_policy(mu=self.means, A = self.constraints, b=self.b)        
        optimal_policy = results["x"]
        
        
        if results["success"] == True:
            # Check neighbors
            w_t_norm = np.matmul(np.matmul(optimal_policy, np.linalg.inv(self.gram_mat)),optimal_policy)
            f_t = 1 + np.sqrt((1/2) * np.log(4/self.delta)
                                        + (len(self.means)/4) * np.log(1 + self.t/len(self.means)))
            hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
            if hash_tuple in self.neighbors:
                neighbors = self.neighbors[hash_tuple]
            else:
                neighbors = compute_neighbors(
                    optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
                )
                self.neighbors[hash_tuple] = neighbors

            # Sample arm 'uniformly'
            arm = self.random_state.choice(self.n_arms, p=self.allocation)

            # Check stopping criterion
            stop = self.stopping_criterion(optimal_policy,arm,f_t,w_t_norm)

            misc = {"allocation": self.allocation, "optimal_policy": optimal_policy}

            return arm, stop, optimal_policy, misc

class ProjectedTnS(TnS):
    """
    Track-n-stop style of algorithm for bandits with linear constraints. That computes the allocation according to normal BAI problem
    """

    def __init__(
        self,
        n_arms,
        A_init,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=True,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts)
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        :param d_tracking: D-tracking or C-tracking
        """
        super().__init__(
            n_arms,
            A_init,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )
        self.unconstrained_neighbors = {}

    def act(self):
        """
        Choose an arm to play
        """
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Compute optimal policy w.r.t. current empirical means
        results1, arm_aux = get_policy(mu=self.means, A=None, b=None)
        results,aux= get_policy(mu=self.means, A = self.constraints, b=self.b)        
        optimal_policy = results["x"]
        optimal_arm = results1["x"]
        
        
        if results["success"] == True and results1["success"] ==True:

        # Check if policy already visited. If yes retrieve neighbors otherwise compute neighbors
            w_t_norm = np.matmul(np.matmul(optimal_policy, np.linalg.inv(self.gram_mat)),optimal_policy)
            f_t = 1 + np.sqrt((1/2) * np.log(4/self.delta)
                                        + (len(self.means)/4) * np.log(1 + self.t/len(self.means)))
            hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
            if not hash_tuple in self.neighbors:
                neighbors = compute_neighbors(
                    optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
                )
                self.neighbors[hash_tuple] = neighbors

            # Check if optimal arm is already visited
            hash_tuple = tuple(optimal_arm.tolist())  # npy not hashable
            if hash_tuple in self.unconstrained_neighbors:
                neighbors = self.unconstrained_neighbors[hash_tuple]
            else:
                neighbors = compute_neighbors(
                    optimal_arm, arm_aux["A"], arm_aux["b"], slack=arm_aux["slack"]
                )
                self.unconstrained_neighbors[hash_tuple] = neighbors

            # Solve game to get allocation
            allocation, game_value = solve_game(
                mu=self.means,
                vertex=optimal_arm,
                neighbors=neighbors,
                dist_type=self.dist_type,
                sigma=self.sigma,
                allocation_A=None,
                allocation_b=None,
            )

            if self.restricted_exploration:
                allocation = project_on_feasible(allocation, self.constraints, self.b)

            # Check if forced exploration is needed else D-tracking
            not_saturated = self.n_pulls < (np.sqrt(self.t) - self.n_arms / 2)
            if not_saturated.any() and self.d_tracking:
                # Play smallest N below sqrt(t) - n_arms/2
                arm = np.argmin(self.n_pulls)

            else:
                # Play arm according to tracking rule
                arm = self.tracking(allocation)

            # Check stopping criterion
            stop = self.stopping_criterion(optimal_policy,arm,f_t,w_t_norm)

            misc = {
                "game_value": game_value,
                "allocation": allocation,
                "optimal_policy": optimal_policy,
            }

            return arm, stop, optimal_policy, misc

########### BANDIT ENVIRONMENTS ############ß


class Bandit:
    """
    Generic bandit class
    """

    def __init__(self, expected_rewards, expected_constraints, seed=None):
        self.n_arms = len(expected_rewards)
        self.expected_rewards = expected_rewards
        self.expected_constraints = expected_constraints
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def sample_mean(self):
        pass

    def get_means(self):
        return self.expected_rewards
    
    def get_constraints(self):
        return self.expected_constraints


class GaussianBandit(Bandit):
    """
    Bandit with gaussian rewards
    """
    def __init__(self, expected_rewards, expected_constraints, seed=None):
        super(GaussianBandit, self).__init__(expected_rewards, expected_constraints, seed)
        #self.noise = noise

    def sample_mean(self):
        return self.random_state.normal(self.expected_rewards,1)
    
    def sample_constraint(self,A):
        B = A.flatten()
        constraints = np.random.normal(B,1)
        return constraints.reshape(A.shape[0],A.shape[1])

class BernoulliBandit(Bandit):
    """
    Bandit with bernoulli rewards
    """

    def __init__(self, expected_rewards, seed=None):
        super(BernoulliBandit, self).__init__(expected_rewards, seed)

    def sample(self):
        return self.random_state.binomial(1, self.expected_rewards)





########### EXPLORATION EXPERIMENT ############

import time

def run_exploration_experiment(bandit, explorer, A, b,mu):
    """
    Run pure-exploration experiment for a given explorer and return stopping time and correctness
    """

    res,_ = get_policy(bandit.get_means(), bandit.get_constraints(), b)
    optimal_policy = res["x"]
    done = False
    t = 0
    running_times = []
    #gram_mat = np.eye(A.shape[1])

    policy_list = []
    done_list = []
    arm_list = []
    simple_regret = []
    constraint_violation = 0
    
    while not done and t<100000:
        t += 1
        # Act
        running_time = time.time()
        try:
            arm, done, policy, log = explorer.act()
            policy_list.append(policy)
            done_list.append(done)
            arm_list.append(arm)
        except TypeError:
            #print(1)
            policy = policy_list[-1]
            arm = arm_list[-1]
            done = done_list[-1]
        #print(policy)

        running_time = time.time() - running_time
        running_times.append(running_time)
        # Observe reward
        reward = bandit.sample_mean()[arm]
        constraint_est = bandit.sample_constraint(A)
        cost = np.matmul(constraint_est,np.eye(A.shape[1])[arm])
        #print(cost)
        if policy is not None:
            simple_regret.append(mu.dot(optimal_policy-policy))
            diff = A.dot(policy) - b
            #print(diff)
            if (diff > 0).sum() > 0:
                constraint_violation+= 1 
        # Update explorer
        explorer.update(arm, reward, cost)


    # Check correctness
    correct = np.array_equal(optimal_policy, policy)

    # Return stopping time, correctness, optimal policy and recommended policy
    return t, correct, optimal_policy, policy, np.mean(running_times), constraint_violation,simple_regret




########### TESTING ############
seed = 1000
if __name__ == "__main__":
    iteration = 500
    mu = np.array([1.5, 1, 1.3, 0.4, 0.3, 0.2])
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0]])
    b = np.array([0.5, 0.5])
    bandit = GaussianBandit(mu,A,seed = seed)
    A_init = bandit.sample_constraint(A)
    stopping_times = []
    reco_policies = []
    constraint_violation_list = []
    eps = []
    
    for iter in range(iteration):
        delta = 0.01
        sol,aux = get_policy(mu, A, b)
        optimal_policy = sol["x"]
        #print(f" Optimal : {optimal_policy}")
        explorer = TnS(
            len(mu),
            A_init = A_init,
            b=b,
            delta=delta,
            restricted_exploration=True,
            dist_type="Gaussian",
        )
        

        t, correct, _, policy, _, constraint_violation,simple_regret = run_exploration_experiment(bandit, explorer, A, b,mu)
        stopping_times.append(t)
        reco_policies.append(policy)
        constraint_violation_list.append(constraint_violation)
        eps.append(np.sqrt(np.sum(np.square(optimal_policy-policy))))
        print(f"Iteration {iter+1} stopped at {t} with recommended policy {policy.round(1)} and constraint was violated {constraint_violation} times")

    with open('TnS_Wlag_easy.txt', 'w') as f:
        f.write(str(stopping_times))
        f.write("\n")
        f.write(str(constraint_violation_list))
        f.write("\n")
        f.write(str(reco_policies))
        f.write("\n")
        f.write(str(simple_regret))
        f.write("\n")
        f.write(str(eps))
    print(f"stopping_times - {np.array(stopping_times)}")
    print(f"Mean stopping time is - {np.mean(np.array(stopping_times))}")
    plt.boxplot(np.array(stopping_times))
    plt.xlabel("TnS-WLag")
    plt.ylabel("Stopping time")
    plt.show()
    plt.plot(np.array(stopping_times), np.array(constraint_violation_list), 'go--', linewidth=2, markersize=12)
    plt.xlabel("Stopping times")
    plt.ylabel("Consraint violation count")
    plt.show()