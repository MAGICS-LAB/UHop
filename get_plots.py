import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

def get_df(data, mode, size):

    if "tiny" in data:
        return pd.read_csv(f'./results/{data}_result_{mode}_{size}_deep.csv')
    else:
        return pd.read_csv(f'./results/{data}_result_{mode}_{size}.csv')


size_range = [1000, 2000, 5000, 10000, 20000, 50000]
# size_range = [50000]

size_range = [1000, 5000, 10000, 20000, 40000, 60000, 80000, 95000]
size_range = [95000]


dataset = ['cifar10', 'cifar100', "tiny_imagenet"]
modes = ['bilevel', 'warmup']

plot_data = {
    'Train Loss':[],
    'Test Loss':[],
    'Train Acc.':[],
    'Test Acc.':[],
    'Epoch':[],
    'Model':[]
}

data = 'tiny_imagenet'
mode = 'warmup'


# def print_results(data):

#     for models in ['Modern Hopfield', 'Sparse Hopfield', 'Modern Hopfield + U-Hop', 'Sparse Hopfield + U-Hop']:
#         for 


def plot_curve(tgt='Train Acc.'):

    for size in size_range:
        plot_data = {
            'Train Loss':[],
            'Test Loss':[],
            'Train Acc.':[],
            'Test Acc.':[],
            'Epoch':[],
            'Model':[]
        }
        df = get_df(data, mode, size)

        acc = []

        for i, row in df.iterrows():
            if row['mode'] == 'MHN+ softmax':
                plot_data['Model'].append('Modern Hopfield')
                plot_data['Train Acc.'].append(row['train acc'])
                plot_data['Test Acc.'].append(row['test acc'])
                plot_data['Train Loss'].append(row['train loss'])
                plot_data['Test Loss'].append(row['test loss'])
                plot_data['Epoch'].append(row['epoch'])
                acc.append(row['test acc'])


            elif row['mode'] == 'MHN+ sparsemax':
                plot_data['Model'].append('Sparse Hopfield')
                plot_data['Train Acc.'].append(row['train acc'])
                plot_data['Test Acc.'].append(row['test acc'])
                plot_data['Train Loss'].append(row['train loss'])
                plot_data['Test Loss'].append(row['test loss'])
                plot_data['Epoch'].append(row['epoch'])

            if row['mode'] == 'UMHN+ softmax':
                plot_data['Model'].append('Modern Hopfield + U-Hop')
                plot_data['Train Acc.'].append(row['train acc'])
                plot_data['Test Acc.'].append(row['test acc'])
                plot_data['Train Loss'].append(row['train loss'])
                plot_data['Test Loss'].append(row['test loss'])
                plot_data['Epoch'].append(row['epoch'])

            elif row['mode'] == 'UMHN+ sparsemax':
                plot_data['Model'].append('Sparse Hopfield + U-Hop')
                plot_data['Train Acc.'].append(row['train acc'])
                plot_data['Test Acc.'].append(row['test acc'])
                plot_data['Train Loss'].append(row['train loss'])
                plot_data['Test Loss'].append(row['test loss'])
                plot_data['Epoch'].append(row['epoch'])


        print(np.max(acc))
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

        sns.set_style('whitegrid')
        sns.lineplot(data=plot_data, x='Epoch', y='Train Acc.', hue='Model', ax=axs[ 0])
        axs[0].set_xlabel('Epoch', fontsize=12)
        axs[0].set_ylabel('Train Acc.', fontsize=11)
        axs[0].set_title('Train Acc.', fontsize=12)
        axs[0].legend(loc='center right', fontsize=11)

        # sns.set_style('whitegrid')
        sns.lineplot(data=plot_data, x='Epoch', y='Test Acc.', hue='Model', ax=axs[1])
        axs[1].set_xlabel('Epoch', fontsize=12)
        axs[1].set_ylabel('Test Acc.', fontsize=11)
        axs[1].set_title('Test Acc.', fontsize=12)
        axs[1].legend(loc='center right', fontsize=11)

        sns.lineplot(data=plot_data, x='Epoch', y='Train Loss', hue='Model', ax=axs[2])
        axs[2].set_xlabel('Epoch', fontsize=12)
        axs[2].set_ylabel('Train Loss', fontsize=11)
        axs[2].set_title('Train Loss', fontsize=12)
        axs[2].legend(loc='center right', fontsize=11)


        sns.lineplot(data=plot_data, x='Epoch', y='Test Loss', hue='Model', ax=axs[3])
        axs[3].set_xlabel('Epoch', fontsize=12)
        axs[3].set_ylabel('Test Loss', fontsize=11)
        axs[3].set_title('Test Loss', fontsize=12)
        axs[3].legend(loc='center right', fontsize=11)

        plt.tight_layout()
        if "tiny" in data:
            
            if size != 95000:
                # plt.title(f'{tgt} Comparison on TinyImageNet (size={size})', fontsize=18)
                plt.savefig(f'./plot_result/TinyImageNet_{mode}_{size}.png', dpi=480)
            else:
                # plt.title(f'{tgt} Comparison on TinyImageNet (size=full)', fontsize=18)
                plt.savefig(f'./plot_result/TinyImageNet_{mode}_{size}.png', dpi=480)

        else:
            if size != 50000:
                # plt.title(f'{tgt} Comparison on {data} (size={size})', fontsize=18)
                plt.savefig(f'./plot_result/{data}_{mode}_{size}.png', dpi=480)
            else:
                # plt.title(f'{tgt} Comparison on {data} (size=full)', fontsize=18)
                plt.savefig(f'./plot_result/{data}_{mode}_{size}.png', dpi=480)
        plt.clf()


def max_train_Acc(tgt='Train Acc.'):

    # for tgt in ['Train Acc.', 'Test Acc.','Train Loss','Test Loss',]

    plot_data = {
        'Train Loss':[],
        'Test Loss':[],
        'Train Acc.':[],
        'Test Acc.':[],
        'Dataset Size':[],
        'Model':[]
    }
    for size in size_range:


        d1, d2 = [], []
        s1, s2 = [], []
        df = get_df(data, mode, size)

        for i, row in df.iterrows():

            if row['epoch'] == 24:

                if row['mode'] == 'MHN+ softmax':
                    plot_data['Model'].append('Modern Hopfield')
                    plot_data['Train Acc.'].append(row['train acc'])
                    plot_data['Test Acc.'].append(row['test acc'])
                    plot_data['Train Loss'].append(row['train loss'])
                    plot_data['Test Loss'].append(row['test loss'])
                    plot_data['Dataset Size'].append(size)
                    print('dense + uhop', row['train acc'], row['test acc'])

                elif row['mode'] == 'MHN+ sparsemax':
                    plot_data['Model'].append('Sparse Hopfield')
                    print('Sparse', row['train acc'], row['test acc'])

                    plot_data['Train Acc.'].append(row['train acc'])
                    plot_data['Test Acc.'].append(row['test acc'])
                    plot_data['Train Loss'].append(row['train loss'])
                    plot_data['Test Loss'].append(row['test loss'])
                    plot_data['Dataset Size'].append(size)

                if row['mode'] == 'UMHN+ softmax':
                    plot_data['Model'].append('Modern Hopfield + U-Hop')
                    print('dense', row['train acc'], row['test acc'])

                    plot_data['Train Acc.'].append(row['train acc'])
                    plot_data['Test Acc.'].append(row['test acc'])
                    plot_data['Train Loss'].append(row['train loss'])
                    plot_data['Test Loss'].append(row['test loss'])
                    plot_data['Dataset Size'].append(size)
                    d1.append( row['train acc'])
                    d2.append( row['test acc'])

                elif row['mode'] == 'UMHN+ sparsemax':
                    plot_data['Model'].append('Sparse Hopfield + U-Hop')
                    plot_data['Train Acc.'].append(row['train acc'])
                    plot_data['Test Acc.'].append(row['test acc'])
                    print('Sparse uhop', row['train acc'], row['test acc'])
                    plot_data['Train Loss'].append(row['train loss'])
                    plot_data['Test Loss'].append(row['test loss'])
                    plot_data['Dataset Size'].append(size)
                    s1.append( row['train acc'])
                    s2.append( row['test acc'])

        print(np.mean(d1), np.std(d1))
        print(np.mean(s1), np.std(s1))

        plt.figure(figsize=(8, 6), dpi=480)
        sns.set_style('whitegrid')
        sns.lineplot(data=plot_data, x='Dataset Size', y=tgt, hue='Model', markers=True)
        plt.ylabel( "Max "+tgt, fontsize=18)
        plt.xlabel('Dataset Size', fontsize=18)
        plt.legend(loc='upper right', fontsize=14)

        plt.title(f'Max {tgt} v.s. Dataset Size ({data})', fontsize=18)
        plt.savefig(f'./plot_result/train_acc_size_{data}_{mode}.png', dpi=480)
        plt.clf()

# plot_curve()

# max_train_Acc()
# for data in ["cifar10", "cifar100"]:
plot_curve()