# Raman dashboard

An interactive [marimo](https://marimo.io) dashboard for Raman microscopy `.rs`
map files. Load a map, extract a composable `RamanHeatmap`, inspect spectra, and
resolve mixture components with **MCR-ALS** — all in the browser.

## Run it

No clone needed — launch straight from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
# with uvx
uvx --from git+https://github.com/MehrResearch/raman_dashboard raman-dashboard

# or with uv run
uv run --from git+https://github.com/MehrResearch/raman_dashboard raman-dashboard
```

This opens the dashboard (`marimo run`) in your browser. A small sample map is
bundled, so it loads to something immediately; use the file browser at the top to
open your own `.rs` file.

Extra arguments are forwarded to `marimo run`, e.g.:

```bash
uvx --from git+https://github.com/MehrResearch/raman_dashboard raman-dashboard --port 2718 --headless
```

## What it does

- Loads `.rs` Raman maps via the bundled [`ramanrs`](src/raman_dashboard/ramanrs.py) reader.
- Pre-processing: Raman-shift range selection, cosmic-ray removal, curved
  fluorescence baseline removal.
- Interactive band heatmap overlaid on the optical image; lasso/box-select points
  to average their spectra; drag across the spectrum to set the heatmap band.
- **MCR-ALS** mixture resolution with adjustable component count and spectral
  sparsity.

## Development

```bash
git clone https://github.com/MehrResearch/raman_dashboard
cd raman_dashboard
uv run raman-dashboard            # run the app
uv run marimo edit src/raman_dashboard/dashboard.py   # edit the notebook
```

## License

MIT
