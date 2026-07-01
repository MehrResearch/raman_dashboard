# Raman dashboard

An interactive [marimo](https://marimo.io) dashboard for Raman microscopy `.rs`
files. Load a map, extract a composable `RamanHeatmap`, inspect spectra, and
resolve mixture components with **MCR-ALS** — all in the browser. Files of
single-point scans (spot measurements) are supported too: the dashboard
switches to a spectra view automatically.

## Run it

Download [uv](https://docs.astral.sh/uv/) then:

```bash
# with uvx
uvx --from git+https://github.com/MehrResearch/raman_dashboard raman-dashboard
```

This opens the dashboard (`marimo run`) in your browser. A small sample map is
bundled, so it loads to something immediately; use the file browser at the top to
open your own `.rs` file.

Extra arguments are forwarded to `marimo run`, e.g.:

```bash
uvx --from git+https://github.com/MehrResearch/raman_dashboard raman-dashboard --port 2718 --headless
```

## What it does

- Loads `.rs` Raman maps and single-point scans via the bundled
  [`ramanrs`](src/raman_dashboard/ramanrs.py) reader.
- Pre-processing: Raman-shift range selection, cosmic-ray removal, curved
  fluorescence baseline removal (shared by maps and scans).
- Interactive band heatmap overlaid on the optical image; lasso/box-select points
  to average their spectra; drag across the spectrum to set the heatmap band.
- **MCR-ALS** mixture resolution with adjustable component count and spectral
  sparsity.
- Scan files show an overlaid, pre-processed spectra view with per-scan metadata.

## Development

```bash
git clone https://github.com/MehrResearch/raman_dashboard
cd raman_dashboard
uv run raman-dashboard            # run the app
uv run marimo edit src/raman_dashboard/dashboard.py   # edit the notebook
```

## License

MIT
