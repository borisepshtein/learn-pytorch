"""View accumulated experiment history from results/experiment_log.jsonl.

Usage:
    python3 view_results.py                                  # show every run
    python3 view_results.py ucsd_ped2_autoencoder             # filter to one script
    python3 view_results.py ucsd_ped2_autoencoder --plot roc_auc
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('script', nargs='?', help='Filter to one script name (e.g. ucsd_ped2_autoencoder)')
    parser.add_argument('--plot', metavar='METRIC', help='Plot METRIC over time and save to results/<script>_<metric>_trend.png')
    args = parser.parse_args()

    df = pd.read_json('results/experiment_log.jsonl', lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if args.script:
        df = df[df['script'] == args.script]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))

    if args.plot:
        if not args.script:
            raise SystemExit('--plot requires a script name so the trend line is a single series')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df[args.plot], marker='o')
        ax.set_xlabel('Run timestamp')
        ax.set_ylabel(args.plot)
        ax.set_title(f'{args.script}: {args.plot} over time')
        fig.autofmt_xdate()

        out_path = f'results/{args.script}_{args.plot}_trend.png'
        fig.savefig(out_path, dpi=150)
        print(f'\nSaved trend plot to {out_path}')


if __name__ == '__main__':
    main()
