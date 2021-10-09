import numpy as np

class DownstreamSector():
    """An array with downstream agents
        
        -- attributes
        A : net worth
        Y : production
        N : number of workers
        Q : amount of requested intermediate goods
        W : wage bill
        u : price of final good
        B : demand of credit
        l : leverage ratio
        rb: interest rate from the bank
        """

    def __init__(self, n_agents=500, phi=1.2, beta=0.8, delta_d=0.5, gamma=0.5, w=1, sigma=0.1, theta=0.5):
        
        # Raise an error if variables are not integers or float
        if not all([any([isinstance(variable,type_) for type_ in [int,float]])\
                    for variable in [phi, beta, delta_d, gamma, w, sigma, theta]]):
            raise TypeError("The arguments phi, delta_d, gamma, w, sigma and theta must be of type int or float")
        if not isinstance(n_agents,int):
            raise TypeError("The argument n must be of type int")
        
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        self.phi = phi
        self.beta = beta
        self.delta_d = delta_d
        self.gamma = gamma
        self.w = w
        self.sigma = sigma
        self.theta = theta
        
    def compute(self):
        "  "
        self.Y = self.phi * pow(self.A,self.beta)                   # Y = phi * A^beta
        self.N = self.delta_d * self.Y                              # N = deltad * Y
        self.Q = self.gamma * self.Y                                # Q = gamma * Y
        self.W = self.w * self.N                                    # W = w * N
        self.u = 2 * np.random.rand(self.n_agents)                  # random price in the range [0,2]
        
    def bank_credit(self, B_net_worth_array = None, DB_connection = None):
        
        # Check if B_net_worth_array and DB_connection are numpy arrays
        if isinstance(B_net_worth_array,np.ndarray) and isinstance(DB_connection,np.ndarray):
            pass
        else:
            raise TypeError("B_net_worth_array and DB_connection arguments must be of type np.ndarray")
        
        # Define the demand of credit 
        A_less_than_W_bool = self.A < self.W                        # True if A < W and False if A >= W
        
        self.B = np.where(A_less_than_W_bool, self.W - self.A, 0)   # (B = W-A if A < W) and (B = 0 if A >= W)
        self.l = np.where(A_less_than_W_bool, self.B/self.A, 0)     # (l = B/A if A < W) and (l = 0 if A >= W)
        
        # Define the interest rate from the bank
        self.rb = np.where(A_less_than_W_bool,\
                         self.sigma * pow(B_net_worth_array[DB_connection],-self.sigma) + self.theta * pow(self.l, self.theta)\
                         , 0)                                       # rb = sigma * Ab^(-theta) + theta * l^theta
        self.rb = np.where(self.rb < 0.001, 0.001, self.rb)         # rb = 0.001 if rb < 0.001
