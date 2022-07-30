import matplotlib.pyplot as plt
import numpy as np

def plot_vars(list_to_plot,title):
    """Plot the variables on 'list_to_plot' versus the time and save them as 'title.pdf' '"""
    fig, axs = plt.subplots(len(list_to_plot), figsize=(20,len(list_to_plot)*3))
    # fig.suptitle(title,fontsize=32)
    for i in range(len(list_to_plot)):
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].plot(list_to_plot[i][0][10:])
        axs[i].set_ylabel(list_to_plot[i][1],fontsize=22)
    plt.savefig(title+".pdf")
    #plt.show()

#def plot_scan(param_list, models_list, title):
#    """Plot  """
def net_worth_means(models_list):
    downstream_net_worth_means=[]
    upstream_net_worth_means=[]
    bank_net_worth_means=[]
    for model in models_list:
        dd=eval('model.d.'+'A'+'_agg')
        downstream_net_worth_means.append(np.array(dd).mean()/model.n_agents_d)
        upstream_net_worth_means.append(np.array(model.u.A_agg).mean()/model.n_agents_u)
        bank_net_worth_means.append(np.array(model.b.A_agg).mean()/model.n_agents_b)
    return downstream_net_worth_means, upstream_net_worth_means, bank_net_worth_means

def compute_means(models_list, variable):
    consumer_means=[]
    downstream_means=[]
    upstream_means=[]
    bank_means=[]
    result={}

    if variable + '_agg' in dir(models_list[0].c):
        for model in models_list:
            cons_agg=eval('model.c.' + variable + '_agg')
            consumer_means.append(np.array(cons_agg).mean()/model.n_agents_c)
        result['consumer']=consumer_means

    if variable + '_agg' in dir(models_list[0].d):
        for model in models_list:
            down_agg=eval('model.d.' + variable + '_agg')
            downstream_means.append(np.array(down_agg).mean()/model.n_agents_d)
        result['downstream']=downstream_means

    if variable + '_agg' in dir(models_list[0].u):
        for model in models_list:
            up_agg=eval('model.u.' + variable + '_agg')
            upstream_means.append(np.array(up_agg).mean()/model.n_agents_u)
        result['upstream']=upstream_means

    if variable + '_agg' in dir(models_list[0].b):
        for model in models_list:
            bank_agg=eval('model.b.' + variable + '_agg')
            bank_means.append(np.array(bank_agg).mean()/model.n_agents_b)
        result['bank']=bank_means
    return result

def plot_variation(param_list, result, xlabel, ylabel, title):
    for k, v in result.items():
        if k=='consumer':
            plt.plot(param_list, v, label=k, color='red')
        if k=='downstream':
            plt.plot(param_list, v, label=k, color=(68/255, 119/255, 170/255))
        if k=='upstream':
            plt.plot(param_list, v, label=k, color='orange')
        if k=='bank':
            plt.plot(param_list, v, label=k, color='green')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(title+".pdf")
    plt.show()
    plt.clf()

def analysis_param(variable, param_list, models_list, xlabel, ylabel, plotdir):
    result = compute_means(models_list, variable)
    plot_variation(param_list, result, xlabel, ylabel, plotdir)
