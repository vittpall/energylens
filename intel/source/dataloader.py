import numpy as np
from scipy.stats.qmc import Sobol
import itertools


class Dataloader:
    """
    Generates configurations using either exhaustive cartesian product, 
    Sobol sampling, or random sampling.
    """

    def __init__(
        self,
        heavy_defs,
        light_defs,
        n_heavy=10,
        n_light=5,
        method="sobol",
        exhaustive=False
    ):
        self.heavy_defs = heavy_defs
        self.light_defs = light_defs
        self.n_heavy = n_heavy
        self.n_light = n_light
        self.method = method
        self.exhaustive = exhaustive

        # Generate configs
        if exhaustive:
            self.heavy_configs = self._expand_all(heavy_defs)
            self.light_configs = self._expand_all(light_defs)
        else:
            self.heavy_configs = self._expand_multiple(heavy_defs, n_heavy)
            self.light_configs = self._expand_multiple(light_defs, n_light)

    # ----------------------------------------------------------
    # Generate ALL exhaustive cartesian product
    # ----------------------------------------------------------
    def _expand_all(self, definitions):
        """Generate all possible cartesian product combinations."""
        final = []
        for defn in definitions:
            keys = list(defn.keys())
            
            # Get all possible values for each key
            ranges = []
            for key in keys:
                val = defn[key]
                if isinstance(val, list):
                    ranges.append([(key, v) for v in val])
                elif isinstance(val, tuple) and len(val) == 2:
                    # For ranges, use all integer values or create samples for float ranges
                    lo, hi = val
                    if isinstance(lo, int) and isinstance(hi, int):
                        samples = list(range(lo, hi + 1))
                    else:
                        # For float ranges, create 11 samples from lo to hi
                        samples = [lo + (hi - lo) * i / 10 for i in range(11)]
                    ranges.append([(key, v) for v in samples])
            
            # Generate all combinations via cartesian product
            if ranges:
                for combo in itertools.product(*ranges):
                    cfg = {k: v for k, v in combo}
                    final.append(cfg)
        
        return final if final else [{}]

    # ----------------------------------------------------------
    def _expand_single(self, definition, n_samples):
        if self.method == "sobol":
            return self._sobol_single(definition, n_samples)
        elif self.method == "random":
            return self._random_single(definition, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    # ----------------------------------------------------------
    # SOBOL sampling (original code)
    # ----------------------------------------------------------
    def _sobol_single(self, definition, n_samples):
        keys = list(definition.keys())
        bounds = [definition[k] for k in keys]

        sob = Sobol(d=len(definition), scramble=True)
        raw = sob.random(n_samples)

        samples = []
        for i in range(n_samples):
            cfg = {}
            for dim, key in enumerate(keys):
                interval = bounds[dim]

                if isinstance(interval, list):
                    idx = int(raw[i, dim] * len(interval))
                    idx = min(idx, len(interval) - 1)
                    cfg[key] = interval[idx]

                else:
                    lo, hi = interval
                    val = lo + raw[i, dim] * (hi - lo)
                    val = int(val) if isinstance(lo, int) and isinstance(hi, int) else float(val)
                    cfg[key] = val

            samples.append(cfg)

        return samples

    # ----------------------------------------------------------
    # RANDOM sampling (NEW)
    # ----------------------------------------------------------
    def _random_single(self, definition, n_samples):
        keys = list(definition.keys())
        bounds = [definition[k] for k in keys]

        samples = []
        for _ in range(n_samples):
            cfg = {}

            for key, interval in zip(keys, bounds):

                # categorical list
                if isinstance(interval, list):
                    cfg[key] = np.random.choice(interval)

                # numeric interval
                else:
                    lo, hi = interval
                    val = np.random.uniform(lo, hi)
                    val = int(val) if isinstance(lo, int) and isinstance(hi, int) else float(val)
                    cfg[key] = val

            samples.append(cfg)

        return samples

    # ----------------------------------------------------------
    # Expand definitions list
    # ----------------------------------------------------------
    def _expand_multiple(self, definitions, n):
        final = []
        for defn in definitions:
            final.extend(self._expand_single(defn, n))
        return final

    # ----------------------------------------------------------
    # Yield Cartesian product of configs
    # ----------------------------------------------------------
    def iterate_configs(self):
        for h in self.heavy_configs:
            for l in self.light_configs:
                yield h, l
