from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

threadsperblock = 32
data = 'nontofsinogram'
sns.set_context('paper')

#-----------------------------------------------------------------------------

df = pd.DataFrame()

# read and analyse the TOF projections
fnames_cpu = sorted(list(Path('results').glob(f'{data}__mode_CPU__*.csv')))
fnames_hybrid = sorted(
    list(
        Path('results').glob(
            f'{data}__mode_hybrid__*__tpb_{threadsperblock}*.csv')))
fnames_gpu = sorted(
    list(
        Path('results').glob(
            f'{data}__mode_GPU__*__tpb_{threadsperblock}*.csv')))

for result_file in (fnames_cpu + fnames_hybrid + fnames_gpu):
    print(result_file.name)
    df = pd.concat((df, pd.read_csv(result_file)))

df['t forward+back (s)'] = df['t forward (s)'] + df['t back (s)']

fig, ax = plt.subplots(3, 3, figsize=(7, 7), sharex=False, sharey='row')

bplot_kwargs = dict(capsize=0.15, errwidth=1.5)

for i, mode in enumerate(['CPU', 'hybrid', 'GPU']):
    df_mode = df.loc[df['mode'] == mode]

    sns.barplot(data=df_mode,
                x='sinogram order',
                y='t forward (s)',
                hue='symmetry axis',
                ax=ax[i, 0],
                **bplot_kwargs)
    sns.barplot(data=df_mode,
                x='sinogram order',
                y='t back (s)',
                hue='symmetry axis',
                ax=ax[i, 1],
                **bplot_kwargs)
    sns.barplot(data=df_mode,
                x='sinogram order',
                y='t forward+back (s)',
                hue='symmetry axis',
                ax=ax[i, 2],
                **bplot_kwargs)

    for j in range(3):
        ax[i, j].set_ylabel(f'{ax[i, j].get_ylabel()} - {mode}')

sns.move_legend(ax[0, 0], "upper right", ncol=2)
for i, axx in enumerate(ax.ravel()):
    axx.grid(ls=':')
    if i > 0:
        axx.get_legend().remove()

# set the same ylim for the GPU rows
gpu_ymax = max([x.get_ylim()[1] for x in ax[1:, 0]])
for i, axx in enumerate(ax[1:, :].ravel()):
    axx.set_ylim(0, gpu_ymax)

fig.tight_layout()
fig.show()