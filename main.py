from model.model import Model

# Instantiate the model with desired parameters
model=Model(periods=1000, n_agents_c=3000, n_agents_d=500, n_agents_u=250, n_agents_b=100,\
                 random_connection_probability=0.01, random_sample_size=5, phi=2, beta=0.9, delta_d=0.5,\
                 delta_u=1, gamma=0.5, w_d=1, w_u=1, sigma=0.1, theta=0.05, alpha=0.5,\
                 tau_s=0.1, w_c=1, qmin=0.1, required_reserve=0.3, r_selic=0.2, r_fraction=0.1, sigma_dep=0.1)
# Run the model
model.run()

# Create a plot with the main variables vs. time
model.plot("plots/plot_name")
