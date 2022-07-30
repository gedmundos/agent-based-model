from func.connections import preferred_partner, preferred_partner_tmp
from func.plots import plot_vars
from sectors.sectors import ConsumerSector, DownstreamSector, UpstreamSector, BankSector

import numpy as np

class Model():
    def __init__(self, periods=1000, n_agents_c=3000, n_agents_d=500, n_agents_u=250, n_agents_b=100,\
                 random_connection_probability=0.01, random_sample_size=5, phi=2, beta=0.9, delta_d=0.5,\
                 delta_u=1, gamma=0.5, w_d=1, w_u=1, sigma=0.1, theta=0.05, alpha=0.5,\
                 tau_s=0.1, w_c=1, qmin=0.1, required_reserve=0.3, r_selic=0.2, r_fraction=0.1, sigma_dep=0.1):
        self.n_agents_c = n_agents_c
        self.n_agents_d = n_agents_d
        self.n_agents_u = n_agents_u
        self.n_agents_b = n_agents_b
        self.periods = periods
        
        self.random_connection_probability = random_connection_probability
        self.random_sample_size = random_sample_size
        self.phi=phi
        self.beta=beta
        self.delta_d=delta_d
        self.delta_u=delta_u
        self.gamma=gamma
        self.w_d=w_d
        self.w_u=w_u
        self.w_c=w_c
        self.sigma=sigma
        self.theta=theta
        self.alpha=alpha
        
        self.qmin=qmin
        self.tau_s=tau_s
        self.required_reserve=required_reserve
        self.r_selic=r_selic
        self.r_fraction=r_fraction
        self.sigma_dep=sigma_dep
        
        self.unemployment_history=[]
        
        ###########################################
        ######### For debugging purposes ##########
        self.vars=[]   
        self.CD_connection_lists=[]
        ###########################################
        
        # Initialize the connections' indices randomly
        self.CD_connection = np.random.randint(n_agents_d,size=n_agents_c).astype(int)
        self.CB_connection = np.random.randint(n_agents_b,size=n_agents_c).astype(int)
        self.DB_connection = np.random.randint(n_agents_b,size=n_agents_d).astype(int)
        self.DU_connection = np.random.randint(n_agents_u,size=n_agents_d).astype(int)
        self.UB_connection = np.random.randint(n_agents_b,size=n_agents_u).astype(int)

        # Instantiate sectors
        self.c = ConsumerSector(n_agents = self.n_agents_c)
        self.d = DownstreamSector(n_agents = self.n_agents_d)
        self.u = UpstreamSector(n_agents = self.n_agents_u)
        self.b = BankSector(n_agents = self.n_agents_b)

        self.c.profit = np.ones(self.c.n_agents)
        self.c.demand_eff_final=np.ones(self.n_agents_c)
        
        ###########################################
        ######### For debugging purposes ##########
        self.d.Y_lists=[]
        self.u.A_lists=[]
        self.u.A1_lists=[]
        self.u.r_lists=[]
        self.d.S_lists=[]
        self.c.demand_lists=[]
        self.c.demand_eff_lists=[]
        self.c.demand_eff_tmp_lists=[]
        self.c.non_satisfied_demand_lists=[]
        self.CD_connection_tmp_lists=[]
        self.indices_non_satisfied_consumers_lists=[]
        self.indices_downstreams_with_production_lists=[]
        self.c.demand_eff_final_lists=[]
        self.c.demand_eff_tmp_large_lists=[]
        self.d.total_demand_eff_tmp_lists=[]
        self.d.total_demand_eff_lists=[]
        self.d.total_demand_eff_1_lists=[]
        ###########################################
        
        
    def run(self):
        for i in range(self.periods):
            #################################
            ## Compute downstream features ##
            #################################
            self.d.Y = self.phi * pow(self.d.A,self.beta) - self.d.S                    # Y = phi * A^beta
            self.d.Y = np.where(self.d.Y<0, 0, self.d.Y)
            self.d.N = self.delta_d * self.d.Y                                # N = deltad * Y
            self.d.Q = self.gamma * self.d.Y                                  # Q = gamma * Y
            self.d.W = self.w_d * self.d.N                                    # W = w * N
            
            #if i==395:
            #    self.dprice0=list(self.d.u)
            
            ######### Compute price ##########
            price_='price_2'
            
            ###############################
            ## Compute consumers' demand ##
            ###############################
            
            self.minimum_cost = self.d.u[self.CD_connection] * self.qmin     # Min_cost = u *qmin
            
            self.c.cond_income = self.c.A >= self.minimum_cost
            self.c.demand = np.where(self.c.cond_income, self.c.A/self.d.u[self.CD_connection], self.qmin)
            
            #######################################################
            ## random consume percentage, fixed number of consumers
            #######################################################
            indices_overconsume = np.random.choice(np.arange(self.n_agents_c), size=int(0.5*self.n_agents_c), replace=False)
            self.c.demand[indices_overconsume]*=1+(np.random.random(len(indices_overconsume)))
            self.c.demand[self.c.demand>500]=500
            #######################################################

            #######################################################################################################
            ############################################## FIRST SALE #############################################
            #######################################################################################################
            ######## Compute effective requested final goods ########
            self.d.total_demand = np.zeros(self.n_agents_d)                    
            np.add.at(self.d.total_demand, self.CD_connection, self.c.demand)  # tot_demand = sum(individual demand)

            self.d.cond_production = self.d.total_demand < self.d.Y + self.d.S
            self.d.total_demand_eff = np.where(self.d.cond_production, self.d.total_demand, self.d.Y + self.d.S)
            
            ######### Update stock #########
            self.d.S = (self.d.Y + self.d.S) - self.d.total_demand_eff         # S = (Y + S) - total_demand_eff          
            ##################
            
            #######################################################
            ######### Compute consumer's effective demand #########
            #######################################################
            demand_fraction=(self.d.total_demand_eff[self.CD_connection]/self.d.total_demand[self.CD_connection])
            self.c.demand_eff = demand_fraction * self.c.demand
            #######################################################################################################
            
            
            
            #######################################################################################################
            ########################################### SECOND SALE ###############################################
            #######################################################################################################
            self.c.non_satisfied_demand = self.c.demand - self.c.demand_eff
            
            indices_downstreams_with_production = np.where(self.d.S>0)
            indices_non_satisfied_consumers = np.where(self.c.non_satisfied_demand>0)
            
            self.CD_connection_tmp = preferred_partner_tmp(self.CD_connection[indices_non_satisfied_consumers], -self.d.u[indices_downstreams_with_production],\
                                            random_connection_probability = self.random_connection_probability,\
                                            random_sample_size=self.random_sample_size)
            ######## Compute effective requested final goods ########
            self.d.total_demand_tmp = np.zeros(len(indices_downstreams_with_production[0]))
            #print("" + str(self.d.total_demand_tmp))
            np.add.at(self.d.total_demand_tmp, self.CD_connection_tmp, self.c.non_satisfied_demand[self.c.non_satisfied_demand>0])  # tot_demand = sum(individual demand)
            
            self.d.cond_production_tmp = self.d.total_demand_tmp < self.d.S[indices_downstreams_with_production]
            self.d.total_demand_eff_tmp = np.where(self.d.cond_production_tmp, self.d.total_demand_tmp, self.d.S[indices_downstreams_with_production])
            
            
            #######################################################
            ######### Compute consumer's effective demand #########
            #######################################################
            # demand_eff = (tot_demand_eff / tot_demand) * demand
            demand_fraction_tmp=(self.d.total_demand_eff_tmp[self.CD_connection_tmp]/self.d.total_demand_tmp[self.CD_connection_tmp])
            self.c.demand_eff_tmp = demand_fraction_tmp * self.c.non_satisfied_demand[indices_non_satisfied_consumers]
            
            self.c.demand_eff_tmp_large = np.zeros(self.n_agents_c)
            np.add.at(self.c.demand_eff_tmp_large, indices_non_satisfied_consumers, self.c.demand_eff_tmp)
            
            ######### Update stock #########
            #self.d.S = self.d.S - self.d.total_demand_eff_tmp         # S = (Y + S) - total_demand_eff
            np.add.at(self.d.S, indices_downstreams_with_production, - self.d.total_demand_eff_tmp)
            
            self.c.demand_eff_final=np.zeros(self.n_agents_c)
            self.c.demand_eff_final+=self.c.demand_eff
            np.add.at(self.c.demand_eff_final, indices_non_satisfied_consumers, self.c.demand_eff_tmp)
            
            np.add.at(self.d.total_demand_eff, indices_downstreams_with_production, self.d.total_demand_eff_tmp)
            
            np.add.at(self.d.total_demand, indices_downstreams_with_production, self.d.total_demand_tmp)
                 
            
            
            #######################################################################################################
            ########################################### THIRD SALE ###############################################
            #######################################################################################################
            self.c.non_satisfied_demand_2 = self.c.demand - self.c.demand_eff -self.c.demand_eff_tmp_large
            
            indices_downstreams_with_production_2 = np.where(self.d.S>0)
            indices_non_satisfied_consumers_2 = np.where(self.c.non_satisfied_demand_2>0)

            if np.array(indices_downstreams_with_production_2).sum()>0:
            
            
            
                self.CD_connection_tmp_2 = preferred_partner_tmp(self.CD_connection[indices_non_satisfied_consumers_2], -self.d.u[indices_downstreams_with_production_2],\
                                            random_connection_probability = self.random_connection_probability,\
                                            random_sample_size=self.random_sample_size)

            ######## Compute effective requested final goods ########
                self.d.total_demand_tmp_2 = np.zeros(len(indices_downstreams_with_production_2[0]))
                np.add.at(self.d.total_demand_tmp_2, self.CD_connection_tmp_2, self.c.non_satisfied_demand_2[self.c.non_satisfied_demand_2>0])  # tot_demand = sum(individual demand)
            
                self.d.cond_production_tmp_2 = self.d.total_demand_tmp_2 < self.d.S[indices_downstreams_with_production_2]
                self.d.total_demand_eff_tmp_2 = np.where(self.d.cond_production_tmp_2, self.d.total_demand_tmp_2, self.d.S[indices_downstreams_with_production_2])
            
            
            #######################################################
            ######### Compute consumer's effective demand #########
            #######################################################
            # demand_eff = (tot_demand_eff / tot_demand) * demand
                demand_fraction_tmp_2=(self.d.total_demand_eff_tmp_2[self.CD_connection_tmp_2]/self.d.total_demand_tmp_2[self.CD_connection_tmp_2])
                self.c.demand_eff_tmp_2 = demand_fraction_tmp_2 * self.c.non_satisfied_demand_2[indices_non_satisfied_consumers_2]
            
            # de#ugging
                self.c.demand_eff_tmp_large_2 = np.zeros(self.n_agents_c)
                np.add.at(self.c.demand_eff_tmp_large_2, indices_non_satisfied_consumers_2, self.c.demand_eff_tmp_2)
            
            ######### Update stock #########
                np.add.at(self.d.S, indices_downstreams_with_production_2, -self.d.total_demand_eff_tmp_2)            
                np.add.at(self.c.demand_eff_final, indices_non_satisfied_consumers_2, self.c.demand_eff_tmp_2)
                np.add.at(self.d.total_demand_eff, indices_downstreams_with_production_2, self.d.total_demand_eff_tmp_2)                        
                np.add.at(self.d.total_demand, indices_downstreams_with_production_2, self.d.total_demand_tmp_2)            
            
            ########################################################
            ##### Compute consumers' bank credits and deposits #####
            ########################################################
            # Define demand of credit for consumers
            # A - u * demand_eff > 0
            net_worth_minus_effective_cost = self.c.A - self.d.u[self.CD_connection]*self.c.demand_eff
            
            # Add second effective cost
            second_effective_cost=self.d.u[indices_downstreams_with_production][self.CD_connection_tmp]*self.c.demand_eff_tmp
            np.add.at(net_worth_minus_effective_cost, indices_non_satisfied_consumers, -second_effective_cost)
            
            # Add third effective cost
            if np.array(indices_downstreams_with_production_2).sum()>0:
                third_effective_cost=self.d.u[indices_downstreams_with_production_2][self.CD_connection_tmp_2]*self.c.demand_eff_tmp_2
                np.add.at(net_worth_minus_effective_cost, indices_non_satisfied_consumers_2, -third_effective_cost)
            
            self.c.afford_bool = net_worth_minus_effective_cost>=0

            self.c.B = np.where(~self.c.afford_bool, -net_worth_minus_effective_cost, 0)  # Credit demand: (B = cost-A if A < cost) and (B = 0 if A >= cost)
                            
            self.c.l = np.where(~self.c.afford_bool, self.c.B/self.c.A, 0)                # (l = B/A if A < cost) and (l = 0 if A >= cost)        
            self.c.deposit = np.where(self.c.afford_bool, net_worth_minus_effective_cost, 0)  # Credit demand: (B = cost-A if A < cost) and (B = 0 if A >= cost)

            # Define the interest rate for bank credit requests
            self.c.rb = np.where(~self.c.afford_bool,\
                            self.sigma * pow(self.b.A[self.CB_connection], -self.sigma) + self.theta * pow(self.c.l, self.theta), 0)
                                                                           # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.c.rb = np.where(self.c.rb < 0.001, 0.001, self.c.rb)               # rb = 0.001 if rb < 0.001
            
            self.c.rb = np.where(np.logical_and(self.c.rb < self.r_selic, self.c.rb>0), self.r_selic, self.c.rb)
            
            # Define the interest rate for deposits
            self.c.r_deposit = np.where(self.c.afford_bool,\
                            self.sigma_dep * pow(self.b.A[self.CB_connection], self.sigma_dep), 0)
                                                                           # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.c.r_deposit = np.where(self.c.r_deposit < 0.001, 0.001, self.c.r_deposit)               # rb = 0.001 if rb < 0.001
            self.c.r_deposit = np.where(np.logical_and(self.c.r_deposit < self.r_fraction*self.r_selic, self.c.r_deposit>0), self.r_fraction*self.r_selic, self.c.r_deposit) 
            
            #############################################
            ######### Compute upstream features #########
            #############################################
            self.u.Q = np.zeros(self.n_agents_u)
            np.add.at(self.u.Q, self.DU_connection,self.d.Q)       # Q = sum(Qd)
            self.u.N = self.delta_u * self.u.Q                     # N = delta_u * Q                                      
            self.u.W = self.w_u * self.u.N                         # W = w_u * N

            ### Define price of intermediate good ####
            self.u.principal_price = np.ones(self.n_agents_u)       
            self.u.r = self.alpha * pow(self.u.A, -self.alpha)     # r = alpha * A_u^(-alpha)
            self.u.r[self.u.r < self.r_selic] = self.r_selic
            self.u.u = self.u.principal_price + self.u.r           # u = 1 + r
            
            #######################################
            ##### Update consumers' net worth #####
            #######################################
            self.c.profit = self.w_c - self.d.u[self.CD_connection]*self.c.demand_eff - self.c.rb*self.c.B + self.c.r_deposit*self.c.deposit
            second_effective_cost=self.d.u[indices_downstreams_with_production][self.CD_connection_tmp]*self.c.demand_eff_tmp
            np.add.at(self.c.profit, indices_non_satisfied_consumers, -second_effective_cost)

            if np.array(indices_downstreams_with_production_2).sum()>0:
                third_effective_cost=self.d.u[indices_downstreams_with_production_2][self.CD_connection_tmp_2]*self.c.demand_eff_tmp_2
                np.add.at(self.c.profit, indices_non_satisfied_consumers_2, -third_effective_cost)
            
            self.c.A += self.c.profit
            random_array=np.random.rand(self.n_agents_c)
            pc=1
            self.c.is_bankrupt = np.logical_and(self.c.A<0.001, random_array<pc)
            
            #########################################
            # Compute bank credit of upstream firms #
            #########################################
            # Define the demand of credit 
            A_less_than_W_bool = self.u.A < self.u.W                                  # True if A < W and False if A >= W
        
            self.u.B = np.where(A_less_than_W_bool, self.u.W - self.u.A, 0)           # (B = W-A if A < W) and (B = 0 if A >= W)
            self.u.l = np.where(A_less_than_W_bool, self.u.B/self.u.A, 0)             # (l = B/A if A < W) and (l = 0 if A >= W)
            
            self.u.deposit = np.where(~A_less_than_W_bool, -self.u.W + self.u.A, 0)   # (deposit = A-W if A > W) and (deposit = 0 if A <= W)
                                        
            # Define the interest rate for bank credit
            self.u.rb = np.where(A_less_than_W_bool,\
                         self.sigma * pow(self.b.A[self.UB_connection],-self.sigma) + self.theta * pow(self.u.l, self.theta)\
                         , 0)                                                 # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.u.rb = np.where(self.u.rb < 0.001, 0.001, self.u.rb)                   # rb = 0.001 if rb < 0.001
            self.u.rb = np.where(np.logical_and(self.u.rb < self.r_selic, self.u.rb>0), self.r_selic, self.u.rb)
                                        
            # Define the interest rate for deposit
            self.u.r_deposit = np.where(A_less_than_W_bool,\
                         self.sigma_dep * pow(self.b.A[self.UB_connection],self.sigma_dep), 0)                                                 # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.u.r_deposit = np.where(self.u.r_deposit < 0.001, 0.001, self.u.r_deposit)      # rb = 0.001 if rb < 0.001
            self.u.r_deposit = np.where(np.logical_and(self.u.r_deposit < self.r_fraction*self.r_selic, self.u.r_deposit>0), self.r_fraction*self.r_selic, self.u.r_deposit)
                
            ###########################################
            # Compute bank credit of downstream firms #
            ###########################################
            
            # Define the demand of credit 
            A_less_than_W_bool = self.d.A < self.d.W                          # True if A < W and False if A >= W
        
            self.d.B = np.where(A_less_than_W_bool, self.d.W - self.d.A, 0)   # (B = W-A if A < W) and (B = 0 if A >= W)
            self.d.l = np.where(A_less_than_W_bool, self.d.B/self.d.A, 0)     # (l = B/A if A < W) and (l = 0 if A >= W)
            
            self.d.deposit = np.where(~A_less_than_W_bool, -self.d.W + self.d.A, 0)   # (B = W-A if A < W) and (B = 0 if A >= W)
            
            # Define the interest rate for bank credit
            self.d.rb = np.where(A_less_than_W_bool,\
                         self.sigma * pow(self.b.A[self.DB_connection],-self.sigma) + self.theta * pow(self.d.l, self.theta)\
                         , 0)                                       # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.d.rb = np.where(self.d.rb < 0.001, 0.001, self.d.rb)         # rb = 0.001 if rb < 0.001
            self.d.rb = np.where(np.logical_and(self.d.rb < self.r_selic, self.d.rb > 0), self.r_selic, self.d.rb)
            
            # Define the interest rate for deposit
            self.d.r_deposit = np.where(A_less_than_W_bool,\
                         self.sigma_dep * pow(self.b.A[self.DB_connection],self.sigma_dep), 0)                                       # rb = sigma * Ab^(-sigma) + theta * l^theta
            #self.d.r_deposit = np.where(self.d.r_deposit < 0.001, 0.001, self.d.r_deposit)         # rb = 0.001 if rb < 0.001
            self.d.r_deposit = np.where(np.logical_and(self.d.r_deposit < self.r_fraction*self.r_selic, self.d.r_deposit>0), self.r_fraction*self.r_selic, self.d.r_deposit) 
            
            
            #############################################################################
            ######## Compute downstreams' profit, bad debt and updated net worth ########
            #############################################################################
            # Compute downstream profit
            #self.d.profit = self.d.u * self.d.Y - self.d.rb * self.d.B - self.d.W - self.u.u[self.DU_connection] * self.d.Q
            self.d.profit = self.d.u * self.d.total_demand_eff - self.d.rb * self.d.B - self.d.W\
                            - self.u.u[self.DU_connection] * self.d.Q + self.d.r_deposit*self.d.deposit
                        
            # Update downstream net worth
            self.d.A += self.d.profit                                      # A = A + profit
            self.d.is_bankrupt = self.d.A <= 0.001                         # A firm is bankrupt if the updated net worth is less than 0.0001
            
            ###############################################################################
            ########### Compute upstream bad debt, profit and updated net worth ###########
            ###############################################################################
            # Define an array with a number of zeros equal to the number of upstream firms.
            self.u.bad_debt = np.zeros(self.n_agents_u)
            # At each array's entry (upstream firm), sum the total amount of intermediate goods requested 
            # by the downstream firms that are bankrupt, times the total price (principal + interest rate).
            np.add.at(self.u.bad_debt,self.DU_connection,np.where(self.d.is_bankrupt, self.u.u[self.DU_connection] * self.d.Q, 0))

            self.u.profit= self.u.u * self.u.Q - self.u.rb * self.u.B - self.u.W + self.u.r_deposit*self.u.deposit
            
            # Update upstream net worth
            self.u.A += self.u.profit - self.u.bad_debt                    # A = A + profit - bad debt                  
            self.u.is_bankrupt = self.u.A <= 0.001                         # A firm is bankrupt if the updated net worth is less than 0.0001            

            ###################################################################
            ###### Compute banks' bad debt, profit and updated net worth ######
            ###################################################################
            
            # Total agents' credit demand for banks
            self.b.total_credit_demand=np.zeros(self.n_agents_b)
            np.add.at(self.b.total_credit_demand, self.CB_connection, self.c.B)
            np.add.at(self.b.total_credit_demand, self.DB_connection, self.d.B)
            np.add.at(self.b.total_credit_demand, self.UB_connection, self.u.B)

            # Total deposit in banks
            self.b.total_deposit=np.zeros(self.n_agents_b)
            np.add.at(self.b.total_deposit, self.CB_connection, self.c.deposit)
            np.add.at(self.b.total_deposit, self.DB_connection, self.d.deposit)
            np.add.at(self.b.total_deposit, self.UB_connection, self.u.deposit)
            
            
            
            ###### Partial profit from credit requests from each sector ######
            self.b.B_C = np.zeros(self.n_agents_b)
            np.add.at(self.b.B_C, self.CB_connection, self.c.rb * self.c.B)     # B_C = sum(r * B) (from consumers)
            self.b.B_D = np.zeros(self.n_agents_b)
            np.add.at(self.b.B_D, self.DB_connection, self.d.rb * self.d.B)     # B_D = sum(r * B) (from downstream)
            self.b.B_U = np.zeros(self.n_agents_b)
            np.add.at(self.b.B_U, self.UB_connection, self.u.rb * self.u.B)     # B_U = sum(r * B) (from upstream)
            
            
            ###### Partial expenditure from deposit to each sector ######
            self.b.deposit_C = np.zeros(self.n_agents_b)
            np.add.at(self.b.deposit_C, self.CB_connection, self.c.r_deposit * self.c.deposit)     # B_C = sum(r * B) (from consumers)
            self.b.deposit_D = np.zeros(self.n_agents_b)
            np.add.at(self.b.deposit_D, self.DB_connection, self.d.r_deposit * self.d.deposit)     # B_D = sum(r * B) (from downstream)
            self.b.deposit_U = np.zeros(self.n_agents_b)
            np.add.at(self.b.deposit_U, self.UB_connection, self.u.r_deposit * self.u.deposit)     # B_U = sum(r * B) (from upstream)
            
            
            self.b.total_money_minus_credit = (self.b.A + (1-self.required_reserve) * self.b.total_deposit) - self.b.total_credit_demand
            self.b.afford_bool = self.b.total_money_minus_credit > 0
            self.b.B = np.where(~self.b.afford_bool, -self.b.total_money_minus_credit, 0)
            self.b.deposit = np.where(self.b.afford_bool, self.b.total_money_minus_credit, 0)
            
            
            ###### Profit ######
            self.b.profit = self.b.B_C + self.b.B_D + self.b.B_U \
                            - self.b.deposit_C - self.b.deposit_D - self.b.deposit_U \
                            + np.random.normal(0,0.08,self.n_agents_b)*self.b.deposit\
                            + self.r_selic*self.required_reserve*self.b.total_deposit - self.r_selic*self.b.total_credit_demand
            
            ###### Bad debt ######
            self.b.bad_debt = np.zeros(self.n_agents_b)
            np.add.at(self.b.bad_debt, self.CB_connection, np.where(self.c.is_bankrupt, (1 + self.c.rb) * self.c.B, 0))
            np.add.at(self.b.bad_debt, self.DB_connection, np.where(self.d.is_bankrupt, (1 + self.d.rb) * self.d.B, 0))
            np.add.at(self.b.bad_debt, self.UB_connection, np.where(self.u.is_bankrupt, (1 + self.u.rb) * self.u.B, 0))
            
            #### Update bank net worth ####
            self.b.A += self.b.profit - self.b.bad_debt                # A = A + profit - bad debt
            self.b.is_bankrupt = self.b.A <= 0.001                     # bank is bankrupt if A <= 0.001

            
            #########################################
            ##### Save some aggregate variables #####
            #########################################
            self.d.S_agg.append(np.sum(self.d.S))
            
                      
            #######################################################################
            ######### Update connections and variables after bankruptcies #########
            #######################################################################
            
            # If firms are connected to bankrupt agents, connect with another random agent
            self.CD_connection = np.where(self.c.is_bankrupt[self.CD_connection], np.random.randint(self.n_agents_d,size=self.n_agents_c), self.CD_connection)
            self.CB_connection = np.where(self.c.is_bankrupt[self.CB_connection], np.random.randint(self.n_agents_b,size=self.n_agents_c), self.CB_connection)
            self.DU_connection = np.where(self.u.is_bankrupt[self.DU_connection], np.random.randint(self.n_agents_u,size=self.n_agents_d), self.DU_connection)
            self.DB_connection = np.where(self.b.is_bankrupt[self.DB_connection], np.random.randint(self.n_agents_b,size=self.n_agents_d), self.DB_connection)
            self.UB_connection = np.where(self.b.is_bankrupt[self.UB_connection], np.random.randint(self.n_agents_b,size=self.n_agents_u), self.UB_connection)

            # Update net_worth(random value in [0,2]) and connections of bankrupt consumers
            self.c.A = np.where(self.c.is_bankrupt, 2*np.random.rand(self.n_agents_c), self.c.A)
            self.CD_connection = np.where(self.c.is_bankrupt, np.random.randint(self.n_agents_d, size=self.n_agents_c), self.CD_connection)
            self.CB_connection = np.where(self.c.is_bankrupt, np.random.randint(self.n_agents_b, size=self.n_agents_c), self.CB_connection)
            
            # Update net_worth(random value in [0,2]) and connections of bankrupt downstream firms
            self.d.A = np.where(self.d.is_bankrupt, 2*np.random.rand(self.n_agents_d), self.d.A)
            self.d.S = np.where(self.d.is_bankrupt, 0, self.d.S)
            
            # Define new price
            self.d.Y = self.phi * pow(self.d.A,self.beta) - self.d.S                    # Y = phi * A^beta
            self.d.Y = np.where(self.d.Y < 0, 0, self.d.Y)
            
            self.d.u = np.where(self.d.is_bankrupt, self.d.u.mean()*(0.75+0.5*np.random.random(self.d.n_agents)), self.d.u)
          
            if price_=='price_1':
                self.d.u += self.tau_s * (self.d.total_demand - (self.d.Y + self.d.S))/2
            if price_=='price_2':
                self.d.u = self.d.u * (1 + self.tau_s * (self.d.total_demand - (self.d.Y + self.d.S))/(self.d.Y + self.d.S))            
            
            self.d.u[self.d.u <= 0.05] = 0.05
            self.d.u[self.d.u > 10] = 10
            
            #self.d.u = np.where(self.d.is_bankrupt, 2*np.random.rand(self.n_agents_d), self.d.u)
            self.d.total_demand = np.where(self.d.is_bankrupt, 0, self.d.total_demand)
            self.DU_connection = np.where(self.d.is_bankrupt, np.random.randint(self.n_agents_u, size = self.n_agents_d), self.DU_connection)
            self.DB_connection = np.where(self.d.is_bankrupt, np.random.randint(self.n_agents_b, size = self.n_agents_d), self.DB_connection)
    
            # Update net_worth(random value in [0,2]) and connections of bankrupt upstream firms
            self.u.A = np.where(self.u.is_bankrupt, 2*np.random.rand(self.n_agents_u), self.u.A)
            self.UB_connection = np.where(self.u.is_bankrupt, np.random.randint(self.n_agents_b, size = self.n_agents_u), self.UB_connection)
    
            # Update net_worth(random value in [0,2]) of bankrupt banks
            self.b.A = np.where(self.b.is_bankrupt, 2*np.random.rand(self.n_agents_b), self.b.A)

            ########################################################################
            ###### Update connections through the preferred-partner algorithm ######
            ########################################################################
            self.CD_connection = preferred_partner(self.CD_connection, -self.d.u, random_connection_probability = self.random_connection_probability, random_sample_size=self.random_sample_size)
            self.CB_connection = preferred_partner(self.CB_connection, self.b.A, random_connection_probability = self.random_connection_probability, random_sample_size = self.random_sample_size);
            self.DU_connection = preferred_partner(self.DU_connection, self.u.A, random_connection_probability = self.random_connection_probability, random_sample_size = self.random_sample_size);
            self.DB_connection = preferred_partner(self.DB_connection, self.b.A, random_connection_probability = self.random_connection_probability, random_sample_size = self.random_sample_size);
            self.UB_connection = preferred_partner(self.UB_connection, self.b.A, random_connection_probability = self.random_connection_probability, random_sample_size = self.random_sample_size);
         
            ######################################
            ###### Save aggregate variables ######
            ######################################
            self.c.append_aggregate_variables()
            self.d.append_aggregate_variables()
            self.u.append_aggregate_variables()
            self.b.append_aggregate_variables()          
            
    def plot(self, filename):
        list_to_plot=[[np.array(self.c.A_agg)/self.n_agents_c, 'Net worth - C'],\
                    [np.array(self.d.A_agg)/self.n_agents_d, 'Net worth - D'],\
                    [np.array(self.u.A_agg)/self.n_agents_u, 'Net worth - U'],\
                    [np.array(self.b.A_agg)/self.n_agents_b, 'Net worth - B'],\
                    [np.array(self.c.profit_agg)/self.n_agents_c, 'Profit - C'],\
                    [np.array(self.d.profit_agg)/self.n_agents_d, 'Profit - D'],\
                    [np.array(self.u.profit_agg)/self.n_agents_u, 'Profit - U'],\
                    [np.array(self.b.profit_agg)/self.n_agents_b, 'Profit - B'],\
                    [np.array(self.c.B_agg)/self.n_agents_c, 'Bank Cred - C'],\
                    [np.array(self.d.B_agg)/self.n_agents_d, 'Bank Cred - D'],\
                    [np.array(self.u.B_agg)/self.n_agents_u, 'Bank Cred - U'],\
                    [np.array(self.c.l_agg)/self.n_agents_c, 'Leverage - C'],\
                    [np.array(self.d.l_agg)/self.n_agents_d, 'Leverage - D'],\
                    [np.array(self.u.l_agg)/self.n_agents_u, 'Leverage - U'],\
                    [np.array(self.u.bad_debt_agg)/self.n_agents_u, 'Bad debt - U'],\
                    [np.array(self.b.bad_debt_agg)/self.n_agents_b, 'Bad debt - B'],\
                    [np.array(self.c.is_bankrupt_agg)/self.n_agents_c, 'Bankrupt - C'],\
                    [np.array(self.d.is_bankrupt_agg)/self.n_agents_d, 'Bankrupt - D'],\
                    [np.array(self.u.is_bankrupt_agg)/self.n_agents_u, 'Bankrupt - U'],\
                    [np.array(self.b.is_bankrupt_agg)/self.n_agents_b, 'Bankrupt - B'],\
                    [np.array(self.d.N_agg)/self.n_agents_d, 'Workers - D'],\
                    [np.array(self.u.N_agg)/self.n_agents_u, 'Workers - U'],\
                    [np.array(self.d.u_agg)/self.n_agents_d, 'Price - D'],\
                    [np.array(self.u.u_agg)/self.n_agents_u, 'Price - U'],\
                    [np.array(self.c.demand_agg)/self.n_agents_c, 'Demand - C'],\
                    [np.array(self.c.demand_eff_final_agg)/self.n_agents_c, 'Eff. demand - C'],\
                    [np.array(self.d.Y_agg)/self.n_agents_d, 'Production - D'],\
                    [np.array(self.d.S_agg)/self.n_agents_d, 'Stock -D']
                    ]

        plot_vars(list_to_plot, filename)
