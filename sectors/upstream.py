import numpy as np

class UpstreamSector():
    """An array with upstream agents."""

    def __init__(self, n_agents=250, alpha=0.1, delta_u=1, w_u=1, sigma=0.1, theta=0.05):
        
        # Raise an error if variables are not integers or float
        if not all([any([isinstance(variable,type_) for type_ in [int,float]])\
                    for variable in [alpha, delta_u, w_u, sigma, theta]]):
            raise TypeError("The arguments alpha, delta_u, gamma, w_u, sigma and theta must be of type int or float")
        if not isinstance(n_agents,int):
            raise TypeError("The argument n must be of type int")
        
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        self.alpha = alpha
        self.delta_u = delta_u
        self.w = w_u
        self.sigma = sigma
        self.theta = theta
        
        self.A_agg = []
        self.N_agg = []
        self.Q_agg = []
        self.W_agg = []
        self.u_agg = []
        self.B_agg = []
        self.l_agg = []
        self.rb_agg = []
        self.profit_agg = []
        self.bad_debt_agg = []
        self.is_bankrupt_agg = []
        
    def compute_firm_features(self, D_requested_intermediate_goods = None, DU_connection = None):
        """Generate the attributes 'Y' (production), 'N' (number of workers), 
            'Q' (number of requested intermediate goods), 'W' (wage bill) and 'u' (price of final goods).
            
            Args:
            D_requested_intermediate_goods (ndarray): An array of shape (number of downstream firms,)
                containing the amount of intermediate goods requested by each downstream firm.
            DU_connection (ndarray): An array of shape (number of downstream firms,) with each entry indicating
                the index of the upstream firm to which each downstream firm is connected.
        """
        
        # Raise an error if D_requested_intermediate_goods or DU_connection are numpy arrays
        if isinstance(D_requested_intermediate_goods,np.ndarray) and isinstance(DU_connection,np.ndarray):
            pass
        elif D_requested_intermediate_goods is None or DU_connection is None:
            raise TypeError("D_requested_intermediate_goods or DU_connection arguments is of NoneType. \
            They both should be of type ndarray")
        else:
            raise TypeError("D_requested_intermediate_goods and DU_connection arguments must be of type ndarray")
        
        self.Q = np.zeros(self.n_agents)
        np.add.at(self.Q, DU_connection,D_requested_intermediate_goods)       # Q = sum(Qd)
        self.N = self.delta_u * self.Q                                        
        self.W = self.w * self.N

        # Define price of intermediate good
        self.principal_price = np.ones(self.n_agents)
        self.r = self.alpha * pow(self.A, -self.alpha)
        self.r[self.r < 0.001] = 0.001
        self.u = self.principal_price + self.r
  
    def compute_bank_credit(self, B_net_worth = None, UB_connection = None):
        """Generate the attributes B(credit demanded to banks), l(leverage) and rb(interest rate for bank credit).
        
        Args:
            B_net_worth (ndarray): An array of shape (number of bank agents,) containing the banks' net worth.
            UB_connection (ndarray): An array of shape (number of upstream firms,) with each entry indicating
                the index of the bank to which each upstream firm is connected.
        """         
        # Raise an error if B_net_worth or UB_connection are not numpy arrays
        if isinstance(B_net_worth,np.ndarray) and isinstance(UB_connection,np.ndarray):
            pass
        elif B_net_worth is None or UB_connection is None:
            raise TypeError("B_net_worth or UB_connection arguments is of NoneType. They both should be of type np.ndarray")
        else:
            raise TypeError("B_net_worth and DB_connection arguments must be of type np.ndarray")
        
        # Define the demand of credit 
        A_less_than_W_bool = self.A < self.W                                  # True if A < W and False if A >= W
        
        self.B = np.where(A_less_than_W_bool, self.W - self.A, 0)             # (B = W-A if A < W) and (B = 0 if A >= W)
        self.l = np.where(A_less_than_W_bool, self.B/self.A, 0)               # (l = B/A if A < W) and (l = 0 if A >= W)
        
        # Define the interest rate from the bank
        self.rb = np.where(A_less_than_W_bool,\
                         self.sigma * pow(B_net_worth[UB_connection],-self.sigma) + self.theta * pow(self.l, self.theta)\
                         , 0)                                                 # rb = sigma * Ab^(-sigma) + theta * l^theta
        self.rb = np.where(self.rb < 0.001, 0.001, self.rb)                   # rb = 0.001 if rb < 0.001

    def compute_profit_and_bad_debt(self, is_bankrupt_D = None, DU_connection = None, D_requested_intermediate_goods = None):
        """Generate the "profit" and "bad_debt" attributes.
        
        Args:
            is_bankrupt_D (ndarray): An array of shape (number of downstream firms,) containing the booleans "True" for each
                bankrupt downstream firm and "False" for each not-bankrupt downstream firm.
            DU_connection (ndarray): An array of shape (number of downstream firms,) with each entry indicating
                the index of the upstream firm to which each downstream firm is connected.
            D_requested_intermediate_goods (ndarray): An array of shape (number of downstream firms,) containing the
                amount of intermediate goods requested by each downstream firm.
        """
        
        # Raise an error if U_price or DU_connection are not numpy arrays
        if isinstance(is_bankrupt_D,np.ndarray) and isinstance(DU_connection,np.ndarray)\
        and isinstance(D_requested_intermediate_goods,np.ndarray):
            pass
        elif is_bankrupt_D is None or DU_connection is None or D_requested_intermediate_goods is None:
            raise TypeError("One of the following arguments is of NoneType: 'is_bankrupt_D', 'DU_connection', 'D_requested_intermediate_goods'. They should be of type ndarray.")
        else:
            raise TypeError("is_bankrupt_D, DU_connection and D_requested_intermediate_goods arguments must be of type ndarray.")
        
        ###### Define bad debt ######
        # Define an array with a number of zeros equal to the number of upstream firms.
        self.bad_debt = np.zeros(self.n_agents)
        # At each array's entry (upstream firm), sum the total amount of intermediate goods requested 
        # by the downstream firms that are bankrupt, times the total price (principal + interest rate).
        np.add.at(self.bad_debt,DU_connection,np.where(is_bankrupt_D, self.u[DU_connection] * D_requested_intermediate_goods, 0))
        
        self.profit= self.u * self.Q - self.rb * self.B - self.W
        
    def update_net_worth(self):
        """Update the firms' net worth and create a 'is_bankrupt' attribute which is 'True' if a firm is bankrupt
            and 'False' if not.""" 
        self.A += self.profit - self.bad_debt                                      
        self.is_bankrupt = self.A <= 0.0001                         # A firm is bankrupt if the updated net worth is less than 0.0001
    
    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        self.A_agg.append(np.sum(self.A))
        self.N_agg.append(np.sum(self.N))
        self.Q_agg.append(np.sum(self.Q))
        self.W_agg.append(np.sum(self.W))
        self.u_agg.append(np.sum(self.u))
        self.B_agg.append(np.sum(self.B))
        self.l_agg.append(np.sum(self.l))
        self.rb_agg.append(np.sum(self.rb))
        self.profit_agg.append(np.sum(self.profit))
        self.bad_debt_agg.append(np.sum(self.bad_debt))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
