import numpy as np

def preferred_partner(link, net_worth, random_connection_probability = 0.01, random_sample_size = 5):
    """ Implement the preferred-partner choice rule given a connection array ('link') 
    and the desired agent's 'net_worth' (A). """
    random_array = np.random.rand(len(link))
    cond_eps = random_array < random_connection_probability
    link = np.where(cond_eps,np.random.randint(len(net_worth),size=len(link)),link)
    
    random_links = np.random.randint(len(net_worth),size=(len(link),random_sample_size))
    random_net_worth = net_worth[random_links]
    
    max_random_net_worth = random_net_worth.max(axis=1)
    max_random_positions = np.argmax(random_net_worth,axis=1)
    max_random_links = np.take_along_axis(random_links, max_random_positions[:,None], axis=1).reshape(len(link),)
    cond = np.logical_and(max_random_net_worth>net_worth[link],~cond_eps)
    link = np.where(cond,max_random_links,link)
    
    return link
