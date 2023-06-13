import matplotlib.pyplot as plt
import csv
import numpy as np

def csv_save(q_table, model_name):
    for outk, outv in q_table.items():
        for ink in range(len(outv)):
            q_table[outk][ink] = np.round(outv[ink],3)
    sorted_q_tabel = sorted(q_table.items())
    with open(f'csv\output-{model_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sorted_q_tabel)

def make_plot(num_MTE, Episode_return, num_epoch, episode_rewards, model_name):
    plt.figure(1)
    plt.xlabel("Num Episode")
    plt.ylabel(f'Mean of {num_MTE} Episode returns')
    plt.plot(Episode_return, label='Episode_return')
    plt.legend()
    plt.savefig(f'picture\{model_name}-Mean of {num_MTE} Episode returns')

    plt.figure(2)
    plt.xlabel("Num Episode")
    plt.ylabel("Episode returns")
    plt.scatter(range(len(episode_rewards)), episode_rewards, s=0.5)
    plt.savefig(f'picture\{model_name}-{num_epoch} Episode returns')

    #plt.show()
    plt.close()