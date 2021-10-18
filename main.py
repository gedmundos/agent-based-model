from sectors.downstream import DownstreamSector
from sectors.upstream import UpstreamSector
from sectors.bank import BankSector
from func.connections import preferred_partner
from func.plots import plot_vars

import numpy as np

n_agents_d = 500
n_agents_u = 250
n_agents_b = 100
periods = 1000

random_connection_probability = 0.01
random_sample_size = 5

# Initialize the connections' indices randomly
DB_connection = np.random.randint(n_agents_b,size=n_agents_d).astype(int)
DU_connection = np.random.randint(n_agents_u,size=n_agents_d).astype(int)
UB_connection = np.random.randint(n_agents_b,size=n_agents_u).astype(int)

# Instantiate sectors
d = DownstreamSector(n_agents = n_agents_d)
u = UpstreamSector(n_agents = n_agents_u)
b = BankSector(n_agents = n_agents_b)

for i in range(periods):
    d.compute_firm_features()

    u.compute_firm_features(D_requested_intermediate_goods = d.Q, DU_connection = DU_connection)
    u.compute_bank_credit(B_net_worth = b.A, UB_connection = UB_connection)

    d.compute_bank_credit(B_net_worth = b.A, DB_connection = DB_connection)
    d.compute_profit(U_price = u.u, DU_connection = DU_connection)
    d.update_net_worth()

    u.compute_profit_and_bad_debt(is_bankrupt_D = d.is_bankrupt, DU_connection = DU_connection,\
                             D_requested_intermediate_goods = d.Q)
    u.update_net_worth()

    b.compute_profit_and_bad_debt(DB_connection = DB_connection, UB_connection = UB_connection, D_bank_interest_rate = d.rb,\
                              U_bank_interest_rate = u.rb, D_bank_credit= d.B, U_bank_credit = u.B,\
                              is_bankrupt_D = d.is_bankrupt, is_bankrupt_U = u.is_bankrupt)
    b.update_net_worth()

    # If firms are connected to bankrupt agents, connect with another random agent
    DU_connection = np.where(u.is_bankrupt[DU_connection], np.random.randint(n_agents_u,size=n_agents_d), DU_connection)
    DB_connection = np.where(b.is_bankrupt[DB_connection], np.random.randint(n_agents_b,size=n_agents_d), DB_connection)
    UB_connection = np.where(b.is_bankrupt[UB_connection], np.random.randint(n_agents_b,size=n_agents_u), UB_connection)
    
    # Update net_worth(random value in [0,2]) and connections of bankrupt downstream firms
    d.A = np.where(d.is_bankrupt, 2*np.random.rand(n_agents_d), d.A)
    DU_connection = np.where(d.is_bankrupt, np.random.randint(n_agents_u,size = n_agents_d), DU_connection)
    DB_connection = np.where(d.is_bankrupt, np.random.randint(n_agents_b,size = n_agents_d), DB_connection)
    
    # Update net_worth(random value in [0,2]) and connections of bankrupt upstream firms
    u.A = np.where(u.is_bankrupt, 2*np.random.rand(n_agents_u), u.A)
    UB_connection = np.where(u.is_bankrupt, np.random.randint(n_agents_b, size = n_agents_u), UB_connection)
    
    # Update net_worth(random value in [0,2]) of bankrupt banks
    b.A = np.where(b.is_bankrupt, 2*np.random.rand(n_agents_b), b.A)
    
    # Update connections through the preferred-partner algorithm
    DU_connection = preferred_partner(DU_connection, u.A, random_connection_probability = random_connection_probability, random_sample_size = random_sample_size);
    DB_connection = preferred_partner(DB_connection, b.A, random_connection_probability = random_connection_probability, random_sample_size = random_sample_size);
    UB_connection = preferred_partner(UB_connection, b.A, random_connection_probability = random_connection_probability, random_sample_size = random_sample_size);
    
    d.append_aggregate_variables()
    u.append_aggregate_variables()
    b.append_aggregate_variables()


list_to_plot=[[np.array(d.A_agg)/n_agents_d, 'Net worth - D'],\
              [np.array(u.A_agg)/n_agents_u, 'Net worth - U'],\
              [np.array(b.A_agg)/n_agents_b, 'Net worth - B'],\
              [np.array(d.profit_agg)/n_agents_d, 'Profit - D'],\
              [np.array(u.profit_agg)/n_agents_d, 'Profit - U'],\
              [np.array(b.profit_agg)/n_agents_d, 'Profit - B'],\
              [np.array(d.B_agg)/n_agents_d, 'Bank Cred - D'],\
              [np.array(u.B_agg)/n_agents_d, 'Bank Cred - U'],\
              [np.array(u.bad_debt_agg)/n_agents_b, 'Bad debt - U'],\
              [np.array(b.bad_debt_agg)/n_agents_b, 'Bad debt - B'],\
              [np.array(d.is_bankrupt_agg)/n_agents_d, 'Bankrupt - D'],\
              [np.array(u.is_bankrupt_agg)/n_agents_u, 'Bankrupt - U'],\
              [np.array(b.is_bankrupt_agg)/n_agents_b, 'Bankrupt - B']
             ]

plot_vars(list_to_plot,"Variables_plots")
