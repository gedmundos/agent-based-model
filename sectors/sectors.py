import numpy as np

class ConsumerSector():
    def __init__(self, n_agents=3000):
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        
        self.A_agg = []
        self.B_agg = []
        self.B_positive_agg = []
        self.l_agg = []
        self.rb_agg = []
        self.rb_positive_agg = []
        self.demand_agg = []
        self.demand_eff_agg = []
        self.demand_eff_final_agg=[]
        self.profit_agg = []
        self.is_bankrupt_agg = []
        self.deposit_agg = []
        self.deposit_positive_agg = []
        self.r_deposit_agg = []
        self.r_deposit_positive_agg = []

        # For debugging
        self.A_lists = []
        self.B_lists = []
        self.l_lists = []
        self.rb_lists = []
        self.profit_lists = []
        self.is_bankrupt_lists = []
        self.demand_lists = []
        self.demand_eff_lists = []

    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        self.A_agg.append(np.sum(self.A))
        self.profit_agg.append(np.sum(self.profit))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
        self.B_agg.append(np.sum(self.B))
        if len(self.B[self.B>0])==0: self.B_positive_agg.append(0)
        else: self.B_positive_agg.append((np.sum(self.B[self.B>0])/len(self.B[self.B>0]))*self.n_agents)
        self.l_agg.append(np.sum(self.l))
        self.rb_agg.append(np.sum(self.rb))
        if len(self.rb[self.B>0])==0: self.rb_positive_agg.append(0)
        else: self.rb_positive_agg.append((np.sum(self.rb[self.B>0])/len(self.rb[self.B>0]))*self.n_agents)
        self.demand_agg.append(np.sum(self.demand))
        self.demand_eff_agg.append(np.sum(self.demand_eff))
        self.demand_eff_final_agg.append(np.sum(self.demand_eff_final))
        self.deposit_agg.append(np.sum(self.deposit))
        if len(self.deposit[self.deposit>0])==0: self.deposit_positive_agg.append(0)
        else: self.deposit_positive_agg.append((np.sum(self.deposit[self.deposit>0])/len(self.deposit[self.deposit>0]))*self.n_agents)
        self.r_deposit_agg.append(self.r_deposit)
        if len(self.r_deposit[self.deposit>0])==0: self.r_deposit_positive_agg.append(0)
        else: self.r_deposit_positive_agg.append((np.sum(self.r_deposit[self.deposit>0])/len(self.r_deposit[self.deposit>0]))*self.n_agents)
    
    # For debugging
    def append_variables_list(self):
        self.A_lists.append(list(self.A))
        self.B_lists.append(list(self.B))
        self.l_lists.append(list(self.l))
        self.rb_lists.append(list(self.rb))
        self.is_bankrupt_lists.append(list(self.is_bankrupt))
        self.demand_lists.append(list(self.demand))
        self.demand_eff_lists.append(list(self.demand_eff))
        self.profit_lists.append(list(self.profit))


class DownstreamSector():
    """An array with downstream agents."""

    def __init__(self, n_agents=500):
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        self.u = np.ones(n_agents)#2 * np.random.rand(self.n_agents)
        self.S = np.zeros(self.n_agents)			# Stock
        self.total_demand = np.zeros(self.n_agents)
        
        self.A_agg = []
        self.Y_agg = []
        self.S_agg = []
        self.N_agg = []
        self.Q_agg = []
        self.W_agg = []
        self.u_agg = []
        self.B_agg = []
        self.B_positive_agg = []
        self.l_agg = []
        self.rb_agg = []
        self.rb_positive_agg = []
        #self.bad_debt_agg = []
        self.profit_agg = []
        self.is_bankrupt_agg = []
        self.deposit_agg = []
        self.deposit_positive_agg = []
        self.r_deposit_agg = []
        self.r_deposit_positive_agg = []

        # For debugging
        self.A_lists=[]
        self.Y_lists=[]
        self.S_lists=[]
        self.N_lists=[]
        self.Q_lists=[]
        self.u_lists=[]
        self.B_lists=[]
        self.l_lists=[]
        self.rb_lists=[]
        #self.bad_debt_lists=[]
        self.profit_lists=[]
        self.is_bankrupt_lists=[]
        self.total_demand_lists=[]
        self.total_demand_eff_lists=[]
        self.W_lists=[]
        
    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        
        self.A_agg.append(np.sum(self.A))
        self.Y_agg.append(np.sum(self.Y))
        #self.S_agg.append(np.sum(self.S))
        self.N_agg.append(np.sum(self.N))
        self.Q_agg.append(np.sum(self.Q))
        self.W_agg.append(np.sum(self.W))
        self.u_agg.append(np.sum(self.u))
        self.B_agg.append(np.sum(self.B))
        if len(self.B[self.B>0])==0: self.B_positive_agg.append(0)
        else: self.B_positive_agg.append((np.sum(self.B[self.B>0])/len(self.B[self.B>0]))*self.n_agents)
        self.l_agg.append(np.sum(self.l))
        self.rb_agg.append(np.sum(self.rb))
        if len(self.rb[self.B>0])==0: self.rb_positive_agg.append(0)
        else: self.rb_positive_agg.append((np.sum(self.rb[self.B>0])/len(self.rb[self.B>0]))*self.n_agents)
        #self.bad_debt_agg.append(np.sum(self.bad_debt))
        self.profit_agg.append(np.sum(self.profit))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
        self.deposit_agg.append(np.sum(self.deposit))
        if len(self.deposit[self.deposit>0])==0: self.deposit_positive_agg.append(0)
        else: self.deposit_positive_agg.append((np.sum(self.deposit[self.deposit>0])/len(self.deposit[self.deposit>0]))*self.n_agents)
        self.r_deposit_agg.append(self.r_deposit)
        if len(self.r_deposit[self.deposit>0])==0: self.r_deposit_positive_agg.append(0)
        else: self.r_deposit_positive_agg.append((np.sum(self.r_deposit[self.deposit>0])/len(self.r_deposit[self.deposit>0]))*self.n_agents)

    def append_variables_list(self):
        self.A_lists.append(list(self.A))
        self.Y_lists.append(list(self.Y))
        self.S_lists.append(list(self.S))
        self.N_lists.append(list(self.N))
        self.Q_lists.append(list(self.Q))
        self.u_lists.append(list(self.u))
        self.B_lists.append(list(self.B))
        self.l_lists.append(list(self.l))
        self.rb_lists.append(list(self.rb))
        #self.bad_debt_lists.append(list(self.bad_debt))
        self.profit_lists.append(list(self.profit))
        self.is_bankrupt_lists.append(list(self.is_bankrupt))
        self.total_demand_lists.append(list(self.total_demand))
        self.total_demand_eff_lists.append(list(self.total_demand_eff))
        self.W_lists.append(list(self.W))

class UpstreamSector():
    """An array with upstream agents."""

    def __init__(self, n_agents=250):
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        
        self.A_agg = []
        self.N_agg = []
        self.Q_agg = []
        self.W_agg = []
        self.u_agg = []
        self.B_agg = []
        self.B_positive_agg = []
        self.l_agg = []
        self.rb_agg = []
        self.rb_positive_agg = []
        self.profit_agg = []
        self.bad_debt_agg = []
        self.is_bankrupt_agg = []
        self.deposit_agg = []
        self.deposit_positive_agg = []
        self.r_deposit_agg = []
        self.r_deposit_positive_agg = []
    
    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        self.A_agg.append(np.sum(self.A))
        self.N_agg.append(np.sum(self.N))
        self.Q_agg.append(np.sum(self.Q))
        self.W_agg.append(np.sum(self.W))
        self.u_agg.append(np.sum(self.u))
        self.B_agg.append(np.sum(self.B))
        if len(self.B[self.B>0])==0: self.B_positive_agg.append(0)
        else: self.B_positive_agg.append((np.sum(self.B[self.B>0])/len(self.B[self.B>0]))*self.n_agents)
        self.l_agg.append(np.sum(self.l))
        self.rb_agg.append(np.sum(self.rb))
        if len(self.rb[self.B>0])==0: self.rb_positive_agg.append(0)
        else: self.rb_positive_agg.append((np.sum(self.rb[self.B>0])/len(self.rb[self.B>0]))*self.n_agents)
        self.profit_agg.append(np.sum(self.profit))
        self.bad_debt_agg.append(np.sum(self.bad_debt))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
        self.deposit_agg.append(np.sum(self.deposit))
        if len(self.deposit[self.deposit>0])==0: self.deposit_positive_agg.append(0)
        else: self.deposit_positive_agg.append((np.sum(self.deposit[self.deposit>0])/len(self.deposit[self.deposit>0]))*self.n_agents)
        self.r_deposit_agg.append(self.r_deposit)
        if len(self.r_deposit[self.deposit>0])==0: self.r_deposit_positive_agg.append(0)
        else: self.r_deposit_positive_agg.append((np.sum(self.r_deposit[self.deposit>0])/len(self.r_deposit[self.deposit>0]))*self.n_agents)


class BankSector():
    def __init__(self, n_agents=100):
        self.n_agents = n_agents
        self.A = np.ones(n_agents)
        self.A_agg = []
        self.profit_agg = []
        self.bad_debt_agg = []
        self.is_bankrupt_agg = []
        self.deposit_agg = []
        self.deposit_positive_agg = []

    def append_aggregate_variables(self):
        """Append the aggregate variables"""
        self.A_agg.append(np.sum(self.A))
        self.profit_agg.append(np.sum(self.profit))
        self.bad_debt_agg.append(np.sum(self.bad_debt))
        self.is_bankrupt_agg.append(np.sum(self.is_bankrupt))
        self.deposit_agg.append(np.sum(self.deposit))
        if len(self.deposit[self.deposit>0])==0: self.deposit_positive_agg.append(0)
        else: self.deposit_positive_agg.append((np.sum(self.deposit[self.deposit>0])/len(self.deposit[self.deposit>0]))*self.n_agents)
