import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_features(dir, index):
    data = pd.read_csv(dir / str(index+'.txt'), sep='\t')
    fig, axs = plt.subplots(5, sharex=True, figsize=(10, 10))
    for i in range(4):
        axs[i].plot(data.iloc[:, i])
        axs[i].set(ylabel=data.columns.values.tolist()[i])
        axs[i].grid()

    yeast_indexes = np.unique(data.iloc[:, 5])
    for yeast in yeast_indexes[np.isnan(yeast_indexes) == False]:
        print(yeast)
        axs[4].plot(data.query('index_nearest == @yeast').loc[:, 'dist_nearest'], label=str(yeast), linestyle='', marker='.')
        axs[4].set(ylabel='nearest yeast')
    axs[4].grid()

    fig.suptitle('Amoeba '+index)
    axs[-1].set(xlabel='frames')
    plt.legend(title='Index of yeast', ncol=2)
    plt.tight_layout()
    plt.savefig(dir / str('Amoeba_'+index+'.png'))
    #plt.show()


