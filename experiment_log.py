import datetime
import json
import os


def log_experiment(script_name, metrics):
    """Append a timestamped record of this run's metrics to results/experiment_log.jsonl.

    Unlike the per-script `*_metrics.json` snapshot (which each script overwrites every
    run), this file accumulates one line per run so past experiments stay comparable
    after code changes.
    """
    os.makedirs('results', exist_ok=True)
    record = {
        'timestamp': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'script': script_name,
        **metrics,
    }
    with open('results/experiment_log.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
