import numpy as np

class DownstreamSector():
    """An array with downstream agents."""

    def __init__(self, n_agents=500, phi=1.2, beta=0.8, delta_d=0.5, gamma=0.5, w_d=1, sigma=0.1, theta=0.05):
        
        # Raise an error if variables are not integers or float
        if not all([any([isinstance(variable,type_) for type_ in [int,float]])\
                    for variable in [phi, beta, delta_d, gamma, w_d, sigma, theta]]):
            raise TypeError("The arguments phi, delta_d, gamma, w_d, sigma and theta must be of type int or float")
        if not isinstance(n_agents,int):
            raise TypeError("The argument n must be of type int")
        
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        self.phi = phi
        self.beta = beta
        self.delta_d = delta_d
        self.gamma = gamma
        self.w = w_d
        self.sigma = sigma
        self.theta = theta
        
        self.A_agg = []
        self.Y_agg = []
        self.N_agg = []
        self.Q_agg = []
        self.W_agg = []
        self.u_agg = []
        self.B_agg = []
        self.l_agg = []
        self.rb_agg = []
        self.profit_agg = []
        self.is_bankrupt_agg = []
        
    def compute_firm_features(self):
        """Generate the attributes 'Y' (production), 'N' (number of workers), 
            'Q' (number of requested intermediate goods), 'W' (wage bill) and 'u' (price of final goods)."""
        
        self.Y = self.phi * pow(self.A,self.beta)                   # Y = phi * A^beta
        self.N = self.delta_d * self.Y                              # N = deltad * Y
        self.Q = self.gamma * self.Y                                # Q = gamma * Y
        self.W = self.w * self.N                                    # W = w * N
        self.u = 2 * np.random.rand(self.n_agents)                  # random price in the range [0,2]
        
    def compute_bank_credit(self, B_net_worth = None, DB_connection = None):
        """Generate the attributes B(credit demanded to banks), l(leverage) and rb(interest rate for bank credit).
        
        Args:
            B_net_worth (ndarray): An array of shape (number of bank agents,) containing the banks' net worth.
            DB_connection (ndarray): An array of shape (number of downstream firms,) with each entry indicating
                the index of the bank to which each downstream firm is connected.
        """
        
        # Raise an error if B_net_worth or DB_connection are not numpy arrays
        if isinstance(B_net_worth,np.ndarray) and isinstance(DB_connection,np.ndarray):
            pass
        elif B_net_worth is None or DB_connection is None:
            raise TypeError("B_net_worth or DB_connection arguments is of NoneType. They both should be of type ndarray.")
        else:
            raise TypeError("B_net_worth and DB_connection arguments must be of type ndarray.")
        
        # Define the demand of credit 
        A_less_than_W_bool = self.A < self.W                        # True if A < W and False if A >= W
        
        self.B = np.where(A_less_than_W_bool, self.W - self.A, 0)   # (B = W-A if A < W) and (B = 0 if A >= W)
        self.l = np.where(A_less_than_W_bool, self.B/self.A, 0)     # (l = B/A if A < W) and (l = 0 if A >= W)
        
        # Define the interest rate from the bank
        self.rb = np.where(A_less_than_W_bool,\
                         self.sigma * pow(B_net_worth[DB_connection],-self.sigma) + self.theta * pow(self.l, self.theta)\
                         , 0)                                       # rb = sigma * Ab^(-sigma) + theta * l^theta
        self.rb = np.where(self.rb < 0.001, 0.001, self.rb)         # rb = 0.001 if rb < 0.001

    def compute_profit(self, U_price = None, DU_connection = None):
        """Generate the profit attribute.
        
        Args:
            U_price (ndarray): An array of shape (number of upstream firms,) containing the intermediate goods'price
                at each upstream firm.
            DU_connection (ndarray): An array of shape (number of downstream firms,) with each entry indicating
                the index of the upstream firm to which each downstream firm is connected."""
        
        # Raise an error if U_price or DU_connection are not numpy arrays
        if isinstance(U_price,np.ndarray) and isinstance(DU_connection,np.ndarray):
            pass
        elif U_price is None or DU_connection is None:
            raise TypeError("U_price or DU_connection arguments is of NoneType. They both should be of type ndarray.")
        else:
            raise TypeError("U_price and DU_connection arguments must be of type ndarray.")
        
        self.profit = self.u * self.Y - self.rb * self.B - self.W - U_price[DU_connection] * self.Q
        
    def update_net_worth(self):
        """Update the firms' net worth and create the 'is_bankrupt' attribute which is 'True' if a firm is bankrupt
            and False if not."""
        
        self.A += self.profit                                       # A = A + profit
        self.is_bankrupt = self.A <= 0.0001                         # A firm is bankrupt if the updated net worth is less than 0.0001
        
    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        
        self.A_agg.append(np.sum(self.A))
        self.Y_agg.append(np.sum(self.Y))
        self.N_agg.append(np.sum(self.N))
        self.Q_agg.append(np.sum(self.Q))
        self.W_agg.append(np.sum(self.W))
        self.u_agg.append(np.sum(self.u))
        self.B_agg.append(np.sum(self.B))
        self.l_agg.append(np.sum(self.l))
        self.rb_agg.append(np.sum(self.rb))
        self.profit_agg.append(np.sum(self.profit))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
