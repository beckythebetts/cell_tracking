import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def results_array(metric, results_path):
    df = pd.read_table(results_path, delimiter='\t')
    th_cells = np.unique(df.loc[:, 'th_cell'])
    th_seeds = np.unique(df.loc[:, 'th_seed'])
    print(len(df.loc[:, metric]))
    results = np.array(df.loc[:, metric]).reshape((len(th_cells), len(th_seeds))).T
    # results = df.
    # for i, th_cell in enumerate(th_cells):
    #     for j, th_seed in enumerate(th_seeds):
    #         results[i, j] = df.at[df.index[(df['th_cell'] == th_cell) & (df['th_seed'] == th_seed)].values[0], metric]
    return results, th_cells, th_seeds

def plot(results_path):
    fig, ax = plt.subplots(2, 2)
    ax_indices = [(i, j) for i in range(2) for j in range(2)]
    #ax_indices = ((0,0), (0, 1), (1,0), (1,1))
    for i, metric in enumerate(['MIOU', 'F1', 'Precision', 'Accuracy']):
        results = results_array(metric, results_path)
        #img = ax[ax_indices[i]].imshow(results[0], extent=(np.min(results[1]), np.max(results[1]), np.min(results[2]), np.max(results[2])), cmap='viridis')
        img = ax[ax_indices[i]].imshow(results[0])
        ax[ax_indices[i]].set_xticks(np.arange(len(results[1])), labels=results[1])
        ax[ax_indices[i]].set_yticks(np.arange(len(results[2])), labels=results[2])
        plt.setp(ax[ax_indices[i]].get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
        plt.colorbar(img, ax=ax[ax_indices[i]])
        ax[ax_indices[i]].set(xlabel='th_cell', ylabel='th_seed')
        ax[ax_indices[i]].set_title(metric)
    plt.tight_layout()
    plt.show()

#plot('challenge_data/20x_withyeast_2D/eval_results.txt')
#plot('results.txt')
#values = [[df.at[i, 'MIOU'] for i in range(len(df.index)) if ]
#grid = np.meshgrid(df.loc[:, 'th_cell'], df.loc[:, 'th_seed'])

