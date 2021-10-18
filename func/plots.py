import matplotlib.pyplot as plt

def plot_vars(list_to_plot,title):
    """Plot the variables on 'list_to_plot versus the time and save them as 'title.pdf' '"""
    fig, axs = plt.subplots(len(list_to_plot), figsize=(20,len(list_to_plot)*3))
    # fig.suptitle(title,fontsize=32)
    for i in range(len(list_to_plot)):
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].plot(list_to_plot[i][0][10:])
        axs[i].set_ylabel(list_to_plot[i][1],fontsize=22)
    plt.savefig(title+".pdf")
    #plt.show()
