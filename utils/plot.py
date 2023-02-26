import os
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='colorblind', font_scale=2.2)
fig_size = (16, 10)


class ProcessResults():
    def __init__(self, base_dir, num_trials: int):
        self.base_dir = base_dir
        self.num_trials = num_trials

    def _get_mean_std(self, rel_path, file_name: str, retrain: bool,
                      x: int = 0, y: int = -1):
        """
        Function to combine the data from mutliple trials.
        rel_path is the relative path from inside the model
        specific folder to the file with name file_name.
        file_name should not include the csv extension.
        """
        data = {}

        for model in ['real', 'quat']:
            first_file = None
            for trial in range(self.num_trials):
                trial += 1
                dir_path = os.path.join(self.base_dir, f'Trial {trial}', model,
                                        rel_path)

                # If data does not exist for a particular trial,
                # return none.
                if not os.path.isdir(dir_path):
                    return None

                if retrain:
                    acc_file = pd.read_csv(
                        os.path.join(dir_path, f'{file_name}_retrain.csv')
                    )
                else:
                    acc_file = pd.read_csv(
                        os.path.join(dir_path, f'{file_name}.csv')
                    )

                if trial == 1:
                    first_file = pd.concat(
                        (acc_file.iloc[:, x], acc_file.iloc[:, y]), axis=1
                    )
                else:
                    agg_len = first_file.shape[0]
                    curr_len = acc_file.shape[0]
                    if curr_len > agg_len:
                        first_file = pd.concat(
                            (first_file, acc_file.iloc[:agg_len, y]),
                            axis=1
                        )
                    elif curr_len < agg_len:
                        first_file = pd.concat(
                            (first_file.iloc[:curr_len], acc_file.iloc[:, y]),
                            axis=1
                        )
                    else:
                        first_file = pd.concat(
                            (first_file, acc_file.iloc[:, y]), axis=1
                        )

            # Get mean and standard deviation.
            mean = first_file.iloc[:, 1:].mean(axis=1)
            std = first_file.iloc[:, 1:].std(axis=1)

            # Combine the data.
            table = first_file.iloc[:, x]
            table = pd.concat((table, mean, std), axis=1)

            data[model] = table

        return data

    def save_model(self):
        for model in ['real', 'quat']:
            file_path = os.path.join(self.base_dir, f'{model}_data.csv')
            self.data[model].to_csv(file_path, index=False)

    def train_log(self, level: int):
        """
        Function to load the train log for Q and R
        where level is the pruning level (n = 0 for no pruning).
        """
        retrain = True
        if level == 0:
            retrain = False

        data = self._get_mean_std(f'Level {level}', 'logger', retrain)

        for model in ['real', 'quat']:
            table = data[model]

            if model == 'real':
                col_names = ['Epoch (R)', 'Mean (R)', 'Std (R)']
            else:
                col_names = ['Epoch (Q)', 'Mean (Q)', 'Std (Q)']

            table.columns = col_names
            data[model] = table

        return data

    def plot_train_log(self, file_name: str):
        """
        Function to plot data from pruning experiments.
        """
        print('Accuracy vs training epochs.')

        data = self.train_log(0)

        plt.figure(figsize=fig_size)
        plt.errorbar(
            x=data['real']['Epoch (R)'],
            y=data['real']['Mean (R)'],
            yerr=data['real']['Std (R)'],
            label='Real',
            fmt='x-.'
        )

        plt.errorbar(
            x=data['quat']['Epoch (Q)'],
            y=data['quat']['Mean (Q)'],
            yerr=data['quat']['Std (Q)'],
            label='Quaternion',
            fmt='v--'
        )

        plt.xlabel('Number of training epochs.')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('images/' + file_name + '_1.png', bbox_inches='tight')
        plt.show()

    def plot_lth(self, file_name: str, start: int = 0, end: int = 2):
        sparsity_data = self.spar_acc_data(True)
        levels = np.arange(start=start, stop=end + 1)

        for model in ['real', 'quat']:
            if model == 'real':
                continue
            model_data = sparsity_data[model]
            print(f'Results for {model}')

            plt.figure(figsize=fig_size)

            for iter in levels:
                data = self.train_log(iter)[model]
                sparsity = model_data.iloc[iter, 0]

                if data is None:
                    break

                plt.errorbar(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    yerr=data.iloc[:, 2],
                    label=f'{sparsity:.0f}%'
                )

            plt.xlabel('Number of training epochs.')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('images/' + file_name + '_0.png', bbox_inches='tight')
            plt.show()

    def spar_acc_data(self, retrain: bool):
        """
        Function to load sparsity vs accuracy data.
        """
        data = self._get_mean_std('', 'acc_data', retrain)

        for model in ['real', 'quat']:
            table = data[model]

            if model == 'real':
                col_names = ['Sparsity (R)', 'Mean (R)', 'Std (R)']
            else:
                col_names = ['Sparsity (Q)', 'Mean (Q)', 'Std (Q)']

            table.columns = col_names

            to_drop = []
            for index, _ in table.iterrows():
                if table.iloc[index, 1] <= 30.0:
                    to_drop.append(index)
            table = table.drop(index=to_drop)

            data[model] = table

        return data

    def plot_spar_acc(self, file_name: str, qr_sparsity=0.25,
                      retrain: bool = True):
        """
        Function to plot data from pruning experiments.
        """
        data = self.spar_acc_data(retrain)

        if retrain:
            last_words = 'when retrained.'
        else:
            last_words = 'during pruning.'
        print(f'Accuracy vs sparsity {last_words}')

        fig, ax = plt.subplots(figsize=fig_size)
        ax.errorbar(
            x=data['real']['Sparsity (R)'],
            y=data['real']['Mean (R)'],
            yerr=data['real']['Std (R)'],
            label='Real',
            fmt='x-.'
        )

        ax.errorbar(
            x=data['quat']['Sparsity (Q)'] * qr_sparsity,
            y=data['quat']['Mean (Q)'],
            yerr=data['quat']['Std (Q)'],
            label='Quaternion',
            fmt='v--'
        )

        ax.set_xscale('log')
        ax.set_xlabel('Percentage of weights left.')
        ax.set_ylabel('Accuracy')
        ax.invert_xaxis()

        if file_name == 'conv_6':
            ax.set_xticks([6.25, 12.5, 25, 50, 100])
        elif file_name == 'conv_6_cifar100':
            ax.set_xticks([3.125, 6.25, 12.5, 25, 50, 100])
        elif file_name == 'conv_4_cifar100':
            ax.set_xticks([3.125, 6.25, 12.5, 25, 50, 100])
        elif file_name == 'lenet_300_100':
            ax.set_xticks([1.5625, 3.125, 6.25, 12.5, 25, 50, 100])
        elif file_name == 'lenet_300_100_cifar10':
            ax.set_xticks([6.25, 12.5, 25, 50, 100])
        else:
            ax.set_xticks([0.78125, 1.5625, 3.125, 6.25, 12.5, 25, 50, 100])

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend()
        plt.savefig('images/' + file_name + '_2.png', bbox_inches='tight')
        plt.show()
