import numpy as np

class BankSector:
    """An array with bank agents."""
    
    def __init__(self, n_agents=100):
        
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        
        self.A_agg = []

        self.profit_agg = []
        self.bad_debt_agg = []
        self.is_bankrupt_agg = []
        
    def compute_profit_and_bad_debt(self, DB_connection = None, UB_connection = None, D_bank_interest_rate = None,\
                                    U_bank_interest_rate = None, D_bank_credit= None, U_bank_credit = None,\
                                    is_bankrupt_D = None, is_bankrupt_U = None):
        
        self.B_D = np.zeros(self.n_agents)
        np.add.at(self.B_D, DB_connection, D_bank_interest_rate * D_bank_credit)
        
        self.B_U = np.zeros(self.n_agents)
        np.add.at(self.B_D, UB_connection, U_bank_interest_rate * U_bank_credit)
        
        self.profit = self.B_D + self.B_U
        
        self.bad_debt = np.zeros(self.n_agents)
        np.add.at(self.bad_debt, DB_connection, np.where(is_bankrupt_D, (1 + D_bank_interest_rate) * D_bank_credit, 0))
        np.add.at(self.bad_debt, UB_connection, np.where(is_bankrupt_U, (1 + U_bank_interest_rate) * U_bank_credit, 0))

    def update_net_worth(self):
        self.A += self.profit - self. bad_debt
        self.is_bankrupt = self.A <= 0.0001
 
    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        self.A_agg.append(np.sum(self.A))
        self.profit_agg.append(np.sum(self.profit))
        self.bad_debt_agg.append(np.sum(self.bad_debt))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
