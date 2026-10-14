"""
Shared --datasets handling for the test scripts.

Every script accepts the same forms, so one dataset list (e.g. in testing.pbs)
can be passed to all of them:

    Label=path/to/profiling_metrics.csv   explicit label (used in plots/CSVs)
    path/to/profiling_metrics.csv         label derived from the run directory
    path/to/run_dir                       folder holding <pattern> (default
                                          profiling_metrics.csv)
    <registry key>                        a name from the script's own DATASETS

With no --datasets, each script keeps its own default (its DATASETS dict, or
auto-discovery for the ranking scripts).
"""

import os

DATASETS_HELP = ('Datasets to evaluate: "Label=path/to/profiling_metrics.csv", '
                 'a CSV path, a run folder, or a key of this script\'s DATASETS '
                 'dict. Default: the script\'s own dataset list.')


def add_datasets_arg(ap, help_text=DATASETS_HELP):
    ap.add_argument('--datasets', nargs='+', default=None, help=help_text)


def path_label(path):
    """Readable label for a bare CSV path: the last two run-directory parts,
    so sibling runs that share a leaf (…/<run>/res_native) stay distinct."""
    parts = os.path.normpath(path).split(os.sep)[:-1]      # drop the file name
    if not parts:
        return os.path.splitext(os.path.basename(path))[0]
    return os.sep.join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _looks_like_path(s):
    return (s.lower().endswith('.csv') or os.sep in s or s.startswith('.')
            or os.path.exists(s))


def resolve_datasets(values, registry, pattern='profiling_metrics.csv'):
    """
    Turn --datasets values into an ordered {label: csv_path} dict.

    values=None returns a copy of `registry`. Unknown registry keys are
    reported and skipped; paths are returned even if the file is missing, so
    each script's own "missing datasets" warning still fires.
    """
    if values is None:
        return dict(registry)
    out, unknown = {}, []
    for v in values:
        label = None
        if '=' in v and not os.path.exists(v):
            label, v = v.split('=', 1)
        if not label and v in registry:
            out[v] = registry[v]
            continue
        if not label and not _looks_like_path(v):
            unknown.append(v)
            continue
        path = os.path.join(v, pattern) if os.path.isdir(v) else v
        out[label or path_label(path)] = path
    if unknown:
        print('[WARNING] Unknown dataset names (not a path, not in this '
              'script\'s DATASETS): ' + ', '.join(unknown))
    return out
