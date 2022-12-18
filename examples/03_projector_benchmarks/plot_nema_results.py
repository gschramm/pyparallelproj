import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('--tpb', type=int, default=32)

args = parser.parse_args()

threadsperblock = args.tpb
res_path = Path('results') / args.dir
sns.set_context('paper')

#-----------------------------------------------------------------------------

df = pd.DataFrame()

# read and analyse the TOF projections
fnames_cpu = sorted(list(res_path.glob(f'nema*__mode_CPU__*.json')))
fnames_hybrid = sorted(
    list(res_path.glob(f'nema*__mode_hybrid__tpb_{threadsperblock}*.json')))
fnames_gpu = sorted(
    list(res_path.glob(f'nema*__mode_GPU__tpb_{threadsperblock}*.json')))

for result_file in (fnames_cpu + fnames_hybrid + fnames_gpu):
    print(result_file.name)
    with open(result_file, 'r') as f:
        df = pd.concat((df, pd.DataFrame(json.load(f))))

df['# events (1e6)'] = df['num_events'] / 1000000

fig, ax = plt.subplots(1, 3, figsize=(7, 7 / 3), sharex=True)

bplot_kwargs = dict(capsize=0.15, errwidth=1.5)

for i, mode in enumerate(['CPU', 'hybrid', 'GPU']):
    df_mode = df.loc[df['mode'] == mode]

    sns.barplot(data=df_mode,
                x='# events (1e6)',
                y='iteration time (s)',
                hue='symmetry axis',
                ax=ax[i],
                **bplot_kwargs)

    ax[i].set_title(mode, fontsize='medium')

for i, axx in enumerate(ax.ravel()):
    axx.grid(ls=':')
    if i > 0:
        axx.get_legend().remove()

fig.tight_layout()
fig.savefig(res_path / 'nema_lm.pdf')
fig.show()

#fig.savefig(res_path / f'{data}_{threadsperblock}.pdf')