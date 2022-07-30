import numpy as np

def preferred_partner(link, variable, random_connection_probability = 0.01, random_sample_size = 5):
    """ Implement the preferred-partner choice rule given a connection array ('link') 
    and the desired agent's 'net_worth' (A). """
    
    # With a probability of "random_connection_probability", the new connection is established randomly
    random_array = np.random.rand(len(link))
    cond_eps = random_array < random_connection_probability
    link = np.where(cond_eps,np.random.randint(len(variable),size=len(link)),link)
    
    # A random sample with "random_sample_size" elements is taken from the "variable" array for each element in the "link" array.
    random_links = np.random.randint(len(variable),size=(len(link),random_sample_size))
    random_variable = variable[random_links]

    # "link" is updated keeping the position of the maximum value of the "variable" sample "random_variable"
    max_random_variable = random_variable.max(axis=1)
    max_random_positions = np.argmax(random_variable,axis=1)
    max_random_links = np.take_along_axis(random_links, max_random_positions[:,None], axis=1).reshape(len(link),)
    cond = np.logical_and(max_random_variable>variable[link],~cond_eps)
    link = np.where(cond,max_random_links,link)
    
    return link

def preferred_partner_tmp(link, variable, random_connection_probability = 0.01, random_sample_size = 5):
    """ Implement the preferred-partner choice rule given a connection array ('link')
    and the desired agent's 'net_worth' (A). """

    # With a probability of "random_connection_probability", the new connection is established randomly
    random_array = np.random.rand(len(link))
    cond_eps = random_array < random_connection_probability
    link = np.where(cond_eps,np.random.randint(len(variable),size=len(link)),link)

    # A random sample with "random_sample_size" elements is taken from the "variable" array for each element in the "link" array.
    random_links = np.random.randint(len(variable),size=(len(link),random_sample_size))
    random_variable = variable[random_links]

    # "link" is updated keeping the position of the maximum value of the "variable" sample "random_variable"
    max_random_variable = random_variable.max(axis=1)
    max_random_positions = np.argmax(random_variable,axis=1)
    max_random_links = np.take_along_axis(random_links, max_random_positions[:,None], axis=1).reshape(len(link),)
    #cond = np.logical_and(max_random_variable>variable[link],~cond_eps)
    link = max_random_links

    return link
