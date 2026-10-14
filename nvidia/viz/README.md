# Paper figures

Run from `nvidia/`. Every figure also writes the CSV of its plotted values.

| Script | Figures | Paper |
|---|---|---|
| `characterization.py` | latency/energy vs (TP, PP), latency-vs-energy optimum, batch size, modality; formula fits (CSV) | Sec. V.B |
| `evaluation.py` | ranking without power | Sec. V.C |

- `datasets.py`: registry of the Zeus datasets (model, architecture family, modality, CSV path, colour). Add a run here once; runs whose CSV does not exist yet are skipped.
- `style.py`: fonts, axis style, and the family styles (dense: squares/solid, MoE: triangles/dashed, SSM: diamonds/dotted, hybrid: pentagons/dash-dotted).
- `legacy/`: device-0-era scripts and their outputs, kept for reference only.

```
python viz/characterization.py --out ../../PDP-paper/images/characterization
```
