import matplotlib.pyplot as plt
import csv
import numpy as np

"""
csv_save: Saving the Q table as a csv file at \csv
Data types(Input):
    q_table: dictionary
    model_name: char                                """
def csv_save(q_table, model_name):
    # Rounding Q value in the Q table
    for outk, outv in q_table.items():
        for ink in range(len(outv)):
            q_table[outk][ink] = np.round(outv[ink],3)

    # Sorting the Q table with a key and inserting column names
    sorted_q_tabel = sorted(q_table.items())
    sorted_q_tabel.insert(0,('state',['Complying Will','Stable Advance','Tough Stand','Miracle Idea']))

    # Saving the Q table as a csv file
    with open(f'csv\output-{model_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in sorted_q_tabel:
            writer.writerow([row[0]] + row[1])

"""
make_plot: making plots of lists and saving figures at \picture
Data types(Input):
    num_MTE, num_epoch: int
    Episode_return, episode_rewards: list
    model_name: char                                            """
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