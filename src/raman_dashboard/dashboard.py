# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import importlib

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns

    import ramanrs

    ramanrs = importlib.reload(ramanrs)
    get_selected_indices, set_selected_indices = mo.state((0,))
    get_band_window, set_band_window = mo.state(None)


    def linear_node_basis(x, nodes):
        """Linear spline basis for fixed spectral node positions."""
        x = np.asarray(x, dtype=float)
        nodes = np.asarray(nodes, dtype=float)
        n_nodes = len(nodes)
        if n_nodes < 2:
            return np.ones((len(x), 1))

        basis = np.zeros((len(x), n_nodes), dtype=float)
        left_index = np.searchsorted(nodes, x, side="right") - 1
        left_index = np.clip(left_index, 0, n_nodes - 2)
        x0 = nodes[left_index]
        x1 = nodes[left_index + 1]
        fraction = (x - x0) / (x1 - x0 + 1e-12)
        rows = np.arange(len(x))
        basis[rows, left_index] = 1.0 - fraction
        basis[rows, left_index + 1] = fraction

        left_edge = x <= nodes[0]
        right_edge = x >= nodes[-1]
        basis[left_edge] = 0.0
        basis[left_edge, 0] = 1.0
        basis[right_edge] = 0.0
        basis[right_edge, -1] = 1.0
        return basis


    def node_baseline(spectra, x, *, n_nodes=5, quantile=0.05, smooth=0.02, iterations=40):
        """Per-spectrum curved baseline from optimised spline-node values."""
        spectra = np.asarray(spectra, dtype=float)
        x = np.asarray(x, dtype=float)
        n_nodes = int(n_nodes)
        nodes = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), n_nodes)
        basis = linear_node_basis(x, nodes)

        nearest_pixels = np.abs(x[:, None] - nodes[None, :]).argmin(axis=0)
        node_values = spectra[:, nearest_pixels].copy()
        coefficients = node_values.copy()

        if n_nodes > 2:
            roughness = np.diff(np.eye(n_nodes), n=2, axis=0)
            roughness_matrix = roughness.T @ roughness
            roughness_norm = np.linalg.norm(roughness_matrix, 2)
        else:
            roughness_matrix = np.zeros((n_nodes, n_nodes))
            roughness_norm = 0.0

        step_size = 0.8 / (
            (np.linalg.norm(basis, 2) ** 2 / len(x))
            + smooth * roughness_norm
            + 1e-9
        )
        for _ in range(int(iterations)):
            baseline = coefficients @ basis.T
            residual = spectra - baseline
            weights = np.where(residual >= 0, quantile, 1.0 - quantile)
            gradient = -((weights * residual) @ basis) / len(x)
            gradient += smooth * coefficients @ roughness_matrix
            coefficients -= step_size * gradient
            coefficients = np.minimum(coefficients, node_values)

        return np.minimum(coefficients @ basis.T, spectra)


    def make_band_heatmap(spectra, x, shape, window, *, agg="mean", cmap="inferno"):
        """Aggregate processed spectra inside a Raman-shift window."""
        spectra = np.asarray(spectra, dtype=float)
        x = np.asarray(x, dtype=float)
        band_low, band_high = sorted(float(v) for v in window)
        band_mask = (x >= band_low) & (x <= band_high)
        if not band_mask.any():
            midpoint = 0.5 * (band_low + band_high)
            band_mask[np.argmin(np.abs(x - midpoint))] = True

        selected_band = spectra[:, band_mask]
        reducers = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "median": np.median,
            "min": np.min,
        }
        values = reducers[agg](selected_band, axis=1)
        return ramanrs.RamanHeatmap(
            values.reshape(shape),
            label=f"{agg} {band_low:g}–{band_high:g} cm$^{{-1}}$",
            unit="counts",
            cmap=cmap,
        )


    def nmf_components(spectra, *, n_components=3, iterations=40, seed=0):
        """Small NumPy NMF by multiplicative updates."""
        data = np.asarray(spectra, dtype=float)
        data = data - np.nanmin(data)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.maximum(data, 0.0) + 1e-9

        rng = np.random.default_rng(seed)
        n_points, n_pixels = data.shape
        n_components = int(n_components)
        weights = rng.random((n_points, n_components)) + 0.1
        components = rng.random((n_components, n_pixels)) + 0.1
        eps = 1e-9

        for _ in range(int(iterations)):
            components *= (weights.T @ data) / (weights.T @ weights @ components + eps)
            weights *= (data @ components.T) / (weights @ (components @ components.T) + eps)

        component_scale = np.maximum(components.max(axis=1, keepdims=True), eps)
        components = components / component_scale
        weights = weights * component_scale.T
        return weights, components


    def crop_grid(spectra, shape, *, rows=slice(None), cols=slice(None)):
        """Crop a row/column region from a flattened map spectra matrix."""
        spectra = np.asarray(spectra)
        grid = spectra.reshape(*shape, spectra.shape[-1])
        index_grid = np.arange(shape[0] * shape[1]).reshape(shape)
        cropped_grid = grid[rows, cols, :]
        cropped_indices = index_grid[rows, cols].ravel()
        return (
            cropped_grid.reshape(-1, cropped_grid.shape[-1]),
            cropped_grid.shape[:2],
            cropped_indices,
        )


    def grid_extent_for_points(raman_map, point_indices):
        """Cell-edge extent for a cropped set of grid points."""
        point_indices = np.asarray(point_indices, dtype=int)
        xs = np.asarray(raman_map.x, dtype=float)[point_indices]
        ys = np.asarray(raman_map.y, dtype=float)[point_indices]
        dx, dy = raman_map.point_separation
        return (
            float(xs.min() - dx / 2),
            float(xs.max() + dx / 2),
            float(ys.min() - dy / 2),
            float(ys.max() + dy / 2),
        )


    def plot_map_overlay(
        raman_map,
        overlay,
        overlay_extent,
        *,
        alpha=0.65,
        cmap=None,
        interpolation="nearest",
        colorbar=False,
    ):
        """Plot an overlay whose grid may be spatially cropped."""
        image_extent = raman_map.image_extent
        image = raman_map.optical_image()
        image_x0, image_x1, image_y0, image_y1 = image_extent
        grid_x0, grid_x1, grid_y0, grid_y1 = overlay_extent

        grid_x0 -= image_x0
        grid_x1 -= image_x0
        grid_y0 -= image_y0
        grid_y1 -= image_y0
        image_x0, image_x1 = 0.0, image_x1 - image_x0
        image_y0, image_y1 = 0.0, image_y1 - image_y0

        fig, ax = plt.subplots(
            figsize=(9, 9 * (image_y1 - image_y0) / (image_x1 - image_x0) + 0.5)
        )
        ax.imshow(image, extent=(image_x0, image_x1, image_y1, image_y0), origin="upper")
        artist = ax.imshow(
            overlay.values,
            extent=(grid_x0, grid_x1, grid_y1, grid_y0),
            origin="upper",
            cmap=cmap or overlay.cmap,
            alpha=alpha,
            interpolation=interpolation,
            zorder=2,
        )
        ax.set(
            xlim=(image_x0, image_x1),
            ylim=(image_y1, image_y0),
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=f"{raman_map.label or 'Raman map'} — {overlay.label}",
        )
        ax.set_aspect("equal")
        if colorbar:
            cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
            cb_label = overlay.label if not overlay.unit else f"{overlay.label} ({overlay.unit})"
            cb.set_label(cb_label)
        raman_map._draw_scalebar(ax, None)
        return fig, ax


    sns.set_theme(
        context="talk",
        style="ticks",
        font="Arial",
        font_scale=0.9,
        rc={"svg.fonttype": "none"},
    )
    return (
        crop_grid,
        get_band_window,
        get_selected_indices,
        grid_extent_for_points,
        make_band_heatmap,
        mo,
        node_baseline,
        np,
        plot_map_overlay,
        plt,
        ramanrs,
        set_band_window,
        set_selected_indices,
    )


@app.cell(hide_code=True)
def _(mo):
    filepicker = mo.ui.file_browser(filetypes=(".rs",), multiple=False)

    mo.vstack(
        [
            mo.md(
                """
                # Raman microscopy dashboard

                Load an `.rs` Raman map, extract a composable `RamanHeatmap`, and inspect spectra from the map.
                """
            ),
            filepicker,
        ],
        gap=1,
    )
    return (filepicker,)


@app.cell
def _(filepicker, ramanrs):
    from pathlib import Path

    if filepicker.value:
        selected_file = filepicker.value[0].path
    else:
        _candidates = sorted(Path.cwd().glob("*.rs")) + sorted(
            Path(__file__).resolve().parent.glob("*.rs")
        )
        selected_file = _candidates[0]
    dataset = ramanrs.read(selected_file)

    dataset
    return dataset, selected_file


@app.cell
def _(dataset, mo, selected_file):
    map_slider = mo.ui.slider(
        start=0,
        stop=len(dataset.maps) - 1,
        value=0,
        step=1,
        label="Map",
    )

    mo.hstack(
        [
            mo.md(
                f"""**File:** `{selected_file.name}`  
    **Dataset:** {dataset.title or dataset.label or 'untitled'}"""
            ),
            map_slider,
        ],
        widths=[3, 1],
        align="center",
    )
    return (map_slider,)


@app.cell
def _(dataset, map_slider, np):
    current_map = dataset.maps[map_slider.value]
    shift = np.asarray(current_map.raman_shift, dtype=float)
    shift_min = float(np.floor(np.nanmin(shift) / 10) * 10)
    shift_max = float(np.ceil(np.nanmax(shift) / 10) * 10)
    default_band_low = max(shift_min, 1550.0)
    default_band_high = min(shift_max, 1650.0)
    if default_band_low >= default_band_high:
        default_band_low, default_band_high = shift_min, min(shift_max, shift_min + 100)
    return (
        current_map,
        default_band_high,
        default_band_low,
        shift_max,
        shift_min,
    )


@app.cell(hide_code=True)
def _(crop_grid, current_map, grid_extent_for_points):
    cropped_spectra, cropped_shape, cropped_point_indices = crop_grid(
        current_map.spectra,
        current_map.shape,
        rows=slice(1, None),
    )
    cropped_grid_extent = grid_extent_for_points(current_map, cropped_point_indices)
    return (
        cropped_grid_extent,
        cropped_point_indices,
        cropped_shape,
        cropped_spectra,
    )


@app.cell(hide_code=True)
def _(
    default_band_high,
    default_band_low,
    get_band_window,
    set_band_window,
    shift_max,
    shift_min,
):
    stored_band_window = get_band_window()
    if stored_band_window is None:
        current_band = (default_band_low, default_band_high)
    else:
        stored_band_low, stored_band_high = sorted(float(v) for v in stored_band_window)
        stored_band_low = max(shift_min, min(shift_max, stored_band_low))
        stored_band_high = max(shift_min, min(shift_max, stored_band_high))
        current_band = (
            (stored_band_low, stored_band_high)
            if stored_band_low < stored_band_high
            else (default_band_low, default_band_high)
        )

    if stored_band_window is None or tuple(float(v) for v in stored_band_window) != current_band:
        set_band_window(current_band)
    return (current_band,)


@app.cell(hide_code=True)
def _(mo, shift_max, shift_min):
    remove_cutoff = min(100.0, shift_max)

    remove_range_checkbox = mo.ui.checkbox(
        value=True,
        label="Remove Raman-shift range before visualisation",
    )
    remove_range_slider = mo.ui.range_slider(
        start=shift_min,
        stop=shift_max,
        value=(max(shift_min, 1000), min(shift_max,4000)),
        step=10,
        label="Range to keep (cm⁻¹)",
    )
    cosmic_checkbox = mo.ui.checkbox(
        value=True,
        label="Remove cosmic-ray spikes",
    )
    cosmic_threshold_slider = mo.ui.slider(
        start=4.0,
        stop=35.0,
        value=25.0,
        step=0.5,
        label="Cosmic spike threshold (MAD)",
    )
    baseline_checkbox = mo.ui.checkbox(
        value=True,
        label="Remove curved fluorescence baseline",
    )
    baseline_nodes_slider = mo.ui.slider(
        start=3,
        stop=10,
        value=5,
        step=1,
        label="Baseline spline nodes",
    )
    return (
        baseline_checkbox,
        baseline_nodes_slider,
        cosmic_checkbox,
        cosmic_threshold_slider,
        remove_range_checkbox,
        remove_range_slider,
    )


@app.cell(hide_code=True)
def _(current_band, mo, set_band_window, shift_max, shift_min):
    band_range = mo.ui.range_slider(
        start=shift_min,
        stop=shift_max,
        value=list(current_band),
        step=10,
        label="Raman shift window (cm⁻¹)",
        on_change=lambda value: set_band_window(tuple(float(v) for v in value)),
    )
    return (band_range,)


@app.cell(hide_code=True)
def _(mo):
    agg_select = mo.ui.dropdown(
        options=["mean", "sum", "max", "median", "min"],
        value="mean",
        label="Aggregate",
    )
    cmap_select = mo.ui.dropdown(
        options=["inferno", "magma", "viridis", "plasma", "cividis"],
        value="inferno",
        label="Colormap",
    )
    alpha_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.65,
        step=0.05,
        label="Overlay opacity",
    )
    interpolation_select = mo.ui.dropdown(
        options=["nearest", "bilinear", "gaussian"],
        value="nearest",
        label="Interpolation",
    )
    return agg_select, alpha_slider, cmap_select, interpolation_select


@app.cell(hide_code=True)
def _(cropped_point_indices, get_selected_indices, mo, set_selected_indices):
    valid_point_indices = set(int(i) for i in cropped_point_indices)
    slider_index = next(
        (int(i) for i in get_selected_indices() if int(i) in valid_point_indices),
        int(cropped_point_indices[0]),
    )
    point_slider = mo.ui.slider(
        start=int(cropped_point_indices[0]),
        stop=int(cropped_point_indices[-1]),
        value=slider_index,
        step=1,
        label="Manual point index",
        on_change=lambda value: set_selected_indices((int(value),)),
    )
    return (point_slider,)


@app.cell(hide_code=True)
def _(mo):
    mcr_checkbox = mo.ui.checkbox(value=True, label="Show MCR-ALS components")
    mcr_k_slider = mo.ui.slider(
        start=2,
        stop=8,
        value=3,
        step=1,
        label="MCR-ALS components (K)",
    )
    mcr_sparsity_slider = mo.ui.slider(
        start=0.0,
        stop=0.08,
        value=0.01,
        step=0.005,
        label="MCR spectral sparsity",
    )
    return mcr_checkbox, mcr_k_slider, mcr_sparsity_slider


@app.cell(hide_code=True)
def _(
    agg_select,
    alpha_slider,
    band_range,
    baseline_checkbox,
    baseline_nodes_slider,
    cmap_select,
    cosmic_checkbox,
    cosmic_threshold_slider,
    current_map,
    interpolation_select,
    mcr_checkbox,
    mcr_k_slider,
    mcr_sparsity_slider,
    mo,
    point_slider,
    remove_range_checkbox,
    remove_range_slider,
):
    controls = mo.vstack(
        [
            mo.md(f"### Controls\n{current_map}"),
            mo.md("**Pre-processing**"),
            remove_range_checkbox,
            remove_range_slider,
            cosmic_checkbox,
            cosmic_threshold_slider,
            baseline_checkbox,
            baseline_nodes_slider,
            mo.md("**Heatmap and spectrum**"),
            band_range,
            agg_select,
            cmap_select,
            alpha_slider,
            interpolation_select,
            point_slider,
            mo.md("**MCR-ALS**"),
            mcr_checkbox,
            mcr_k_slider,
            mcr_sparsity_slider,
            mo.md(
                "Pre-processing is applied before the heatmap, spectrum, and MCR-ALS. "
                "Drag a box/lasso over the heatmap to average points; drag across the spectrum to set the Raman band."
            ).callout(kind="info"),
        ],
        gap=0.65,
    )
    return (controls,)


@app.cell(hide_code=True)
def _(mcr_k_slider, mo):
    mcr_component_slider = mo.ui.slider(
        start=1,
        stop=mcr_k_slider.value,
        value=1,
        step=1,
        label="MCR component to view",
    )

    mcr_component_control = mo.vstack([mcr_component_slider], gap=0.4)
    return mcr_component_control, mcr_component_slider


@app.cell(hide_code=True)
def _(
    baseline_checkbox,
    baseline_nodes_slider,
    cosmic_checkbox,
    cosmic_threshold_slider,
    cropped_shape,
    cropped_spectra,
    current_map,
    mo,
    node_baseline,
    np,
    ramanrs,
    remove_range_checkbox,
    remove_range_slider,
):
    pre_raw_axis = np.asarray(current_map.raman_shift, dtype=float)
    pre_raw_spectra = cropped_spectra.astype(float)

    if remove_range_checkbox.value:
        remove_low, remove_high = sorted(float(v) for v in remove_range_slider.value)
        pre_keep_mask = ((pre_raw_axis >= remove_low) & (pre_raw_axis <= remove_high))
    else:
        remove_low, remove_high = None, None
        pre_keep_mask = np.ones_like(pre_raw_axis, dtype=bool)

    viz_axis = pre_raw_axis[pre_keep_mask]
    viz_spectra = pre_raw_spectra[:, pre_keep_mask]

    if cosmic_checkbox.value:
        viz_spectra, cosmic_mask = ramanrs.remove_cosmics(
            viz_spectra,
            threshold=cosmic_threshold_slider.value,
            window=3,
            iterations=2,
            return_mask=True,
        )
    else:
        cosmic_mask = np.zeros(viz_spectra.shape, dtype=bool)

    if baseline_checkbox.value:
        viz_baseline = node_baseline(
            viz_spectra,
            viz_axis,
            n_nodes=baseline_nodes_slider.value,
        )
        viz_spectra = np.maximum(viz_spectra - viz_baseline, 0.0)
    else:
        viz_baseline = None

    removed_text = (
        f"removed `{remove_low:g}`–`{remove_high:g}` cm⁻¹"
        if remove_range_checkbox.value
        else "range removal disabled"
    )
    cosmic_text = (
        f"cosmic removal replaced `{int(cosmic_mask.sum())}` pixels"
        if cosmic_checkbox.value
        else "cosmic removal disabled"
    )
    baseline_text = (
        f"baseline removed with `{baseline_nodes_slider.value}` spline nodes"
        if baseline_checkbox.value
        else "baseline removal disabled"
    )
    preprocess_summary = mo.md(
        f"""
        **Pre-processing:** top grid row removed first, leaving `{cropped_shape[0]}`×`{cropped_shape[1]}` points; {removed_text}; {cosmic_text}; {baseline_text}.  
        Spectrum pixels shown: `{viz_spectra.shape[1]}` / `{current_map.spectra.shape[1]}`.
        """
    ).callout(kind="neutral")
    return pre_keep_mask, preprocess_summary, viz_axis, viz_spectra


@app.cell
def _(
    agg_select,
    alpha_slider,
    cmap_select,
    cropped_grid_extent,
    cropped_point_indices,
    cropped_shape,
    current_band,
    current_map,
    interpolation_select,
    make_band_heatmap,
    mo,
    np,
    plot_map_overlay,
    plt,
    viz_axis,
    viz_spectra,
):
    band_overlay = make_band_heatmap(
        viz_spectra,
        viz_axis,
        cropped_shape,
        current_band,
        agg=agg_select.value,
        cmap=cmap_select.value,
    )
    band_image_x0, _, band_image_y0, _ = current_map.image_extent
    point_x = np.asarray(current_map.x, dtype=float)[cropped_point_indices] - band_image_x0
    point_y = np.asarray(current_map.y, dtype=float)[cropped_point_indices] - band_image_y0

    fig_map, ax_map = plot_map_overlay(
        current_map,
        band_overlay,
        cropped_grid_extent,
        alpha=alpha_slider.value,
        cmap=cmap_select.value,
        interpolation=interpolation_select.value,
    )
    heatmap_selection = mo.ui.matplotlib(ax_map, debounce=True)
    plt.close(fig_map)

    heatmap_panel = mo.vstack(
        [
            mo.md("### Heatmap overlay"),
            heatmap_selection,
        ],
        gap=0.6,
    )
    return heatmap_panel, heatmap_selection, point_x, point_y


@app.cell
def _(
    cropped_point_indices,
    heatmap_selection,
    np,
    point_x,
    point_y,
    set_selected_indices,
):
    try:
        selection_mask = (
            heatmap_selection.value.get_mask(point_x, point_y)
            if heatmap_selection.value
            else None
        )
    except Exception:
        selection_mask = None

    if selection_mask is not None:
        picked_positions = np.flatnonzero(selection_mask)
        picked_indices = tuple(int(cropped_point_indices[i]) for i in picked_positions)
        if picked_indices:
            set_selected_indices(picked_indices)
    return


@app.cell
def _(
    cropped_point_indices,
    current_map,
    get_selected_indices,
    mo,
    np,
    viz_spectra,
):
    cropped_index_lookup = {
        int(point_index): position
        for position, point_index in enumerate(cropped_point_indices)
    }
    raw_selected_indices = tuple(int(i) for i in get_selected_indices())
    selected_indices = tuple(
        i for i in raw_selected_indices if i in cropped_index_lookup
    ) or (int(cropped_point_indices[0]),)
    selected_positions = [cropped_index_lookup[i] for i in selected_indices]
    selected_count = len(selected_indices)
    selected_spectra = viz_spectra[selected_positions].astype(float)
    average_counts = selected_spectra.mean(axis=0)

    selected_rows = [divmod(i, current_map.shape[1])[0] for i in selected_indices]
    selected_cols = [divmod(i, current_map.shape[1])[1] for i in selected_indices]
    selected_xs = np.asarray(current_map.x, dtype=float)[list(selected_indices)]
    selected_ys = np.asarray(current_map.y, dtype=float)[list(selected_indices)]

    if selected_count == 1:
        location = (
            f"point `{selected_indices[0]}` · row `{selected_rows[0]}` · col `{selected_cols[0]}`  \n"
            f"stage x/y: `{selected_xs[0]:.2f}`, `{selected_ys[0]:.2f}` µm"
        )
    else:
        preview = ", ".join(str(i) for i in selected_indices[:8])
        if selected_count > 8:
            preview += ", …"
        location = (
            f"indices: `{preview}`  \n"
            f"rows `{min(selected_rows)}`–`{max(selected_rows)}` · cols `{min(selected_cols)}`–`{max(selected_cols)}`  \n"
            f"mean stage x/y: `{selected_xs.mean():.2f}`, `{selected_ys.mean():.2f}` µm"
        )

    selected_summary = mo.md(
        f"""
        **Selected spectra:** `{selected_count}` point{'s' if selected_count != 1 else ''}  
        {location}
        """
    ).callout(kind="success")
    return average_counts, selected_count, selected_spectra, selected_summary


@app.cell
def _(
    average_counts,
    current_band,
    mo,
    np,
    plt,
    selected_count,
    selected_spectra,
    selected_summary,
    viz_axis,
):
    x_axis = viz_axis
    fig_spec, ax_spec = plt.subplots(figsize=(8, 4))
    if selected_count > 1:
        q10, q90 = np.percentile(selected_spectra, [10, 90], axis=0)
        ax_spec.fill_between(
            x_axis,
            q10,
            q90,
            color="0.7",
            alpha=0.25,
            linewidth=0,
            label="10–90% of selected spectra",
        )
    ax_spec.plot(x_axis, average_counts, color="black", linewidth=1.2, label="average spectrum")
    spectrum_band_lo, spectrum_band_hi = sorted(float(v) for v in current_band)
    ax_spec.axvspan(spectrum_band_lo, spectrum_band_hi, color="crimson", alpha=0.18, label="heatmap band")
    ax_spec.axhline(
        float(np.median(average_counts)),
        color="0.5",
        linewidth=0.8,
        linestyle="--",
        label="median",
    )
    ax_spec.set(
        xlabel="Raman shift (cm$^{-1}$)",
        ylabel="Processed counts",
        title=f"Average spectrum over {selected_count} point{'s' if selected_count != 1 else ''}",
        xlim=(float(np.nanmin(x_axis)), float(np.nanmax(x_axis))),
    )
    ax_spec.legend(loc="upper right", frameon=False)
    fig_spec.tight_layout()
    spectrum_selection = mo.ui.matplotlib(ax_spec, debounce=True)
    plt.close(fig_spec)

    spectrum_panel = mo.vstack(
        [
            mo.md("### Spectrum"),
            selected_summary,
            spectrum_selection,
            mo.md("Drag over the spectrum to choose the Raman band used in the heatmap."),
        ],
        gap=0.6,
    )
    return spectrum_panel, spectrum_selection, x_axis


@app.cell(hide_code=True)
def _(current_band, np, set_band_window, spectrum_selection, x_axis):
    spectrum_band_selection = spectrum_selection.value
    if spectrum_band_selection:
        if hasattr(spectrum_band_selection, "x_min"):
            selected_band_lo = spectrum_band_selection.x_min
            selected_band_hi = spectrum_band_selection.x_max
        elif hasattr(spectrum_band_selection, "vertices"):
            selected_band_vertex_x = [xy[0] for xy in spectrum_band_selection.vertices]
            selected_band_lo = min(selected_band_vertex_x)
            selected_band_hi = max(selected_band_vertex_x)
        else:
            selected_band_lo = None
            selected_band_hi = None

        if selected_band_lo is not None and selected_band_hi is not None:
            selected_axis_lo = float(np.nanmin(x_axis))
            selected_axis_hi = float(np.nanmax(x_axis))
            selected_band_lo, selected_band_hi = sorted((float(selected_band_lo), float(selected_band_hi)))
            selected_band_lo = max(selected_axis_lo, min(selected_axis_hi, selected_band_lo))
            selected_band_hi = max(selected_axis_lo, min(selected_axis_hi, selected_band_hi))
            selected_band_window = (selected_band_lo, selected_band_hi)
            if selected_band_hi > selected_band_lo and any(
                abs(a - b) > 1e-6 for a, b in zip(selected_band_window, current_band)
            ):
                set_band_window(selected_band_window)
    return


@app.cell(hide_code=True)
def _(
    alpha_slider,
    cmap_select,
    cropped_grid_extent,
    cropped_shape,
    current_map,
    interpolation_select,
    mcr_checkbox,
    mcr_component_slider,
    mcr_k_slider,
    mcr_sparsity_slider,
    mo,
    np,
    plot_map_overlay,
    plt,
    ramanrs,
    viz_axis,
    viz_spectra,
):
    if mcr_checkbox.value:
        mcr_result = ramanrs.mcr_als(
            viz_spectra,
            n_components=mcr_k_slider.value,
            n_iter=80,
            spectral_sparsity=mcr_sparsity_slider.value,
            axis=viz_axis,
            shape=cropped_shape,
        )
        mcr_component_index = int(mcr_component_slider.value) - 1
        mcr_component_index = int(np.clip(mcr_component_index, 0, mcr_result.n_components - 1))
        mcr_overlay = mcr_result.heatmap(mcr_component_index, cmap=cmap_select.value)
        mcr_fig_map, mcr_ax_map = plot_map_overlay(
            current_map,
            mcr_overlay,
            cropped_grid_extent,
            alpha=alpha_slider.value,
            cmap=cmap_select.value,
            interpolation=interpolation_select.value,
        )
        mcr_ax_map.set_title(f"MCR-ALS component {mcr_component_index + 1} spatial loading")
        plt.close(mcr_fig_map)

        mcr_fig_spec, mcr_ax_spec = plt.subplots(figsize=(8, 3.3))
        for mcr_loop_index in range(mcr_result.n_components):
            mcr_ax_spec.plot(
                viz_axis,
                mcr_result.spectra[mcr_loop_index],
                linewidth=1.0 if mcr_loop_index != mcr_component_index else 2.0,
                alpha=0.35 if mcr_loop_index != mcr_component_index else 1.0,
                label=f"C{mcr_loop_index + 1}",
            )
        mcr_ax_spec.set(
            xlabel="Raman shift (cm$^{-1}$)",
            ylabel="Component spectrum (normalised)",
            title=f"MCR-ALS spectra · relative error {mcr_result.reconstruction_error:.3f}",
            xlim=(float(np.nanmin(viz_axis)), float(np.nanmax(viz_axis))),
        )
        mcr_ax_spec.legend(frameon=False, ncol=min(4, mcr_result.n_components))
        mcr_fig_spec.tight_layout()
        plt.close(mcr_fig_spec)

        mcr_panel = mo.vstack(
            [
                mo.md("### MCR-ALS mixture components"),
                mo.hstack([mcr_fig_map, mcr_fig_spec], widths=[1, 1], gap=1),
                mo.md(
                    "MCR-ALS uses non-negative ALS with pure-pixel initialisation and a small spectral sparsity constraint."
                ).callout(kind="info"),
            ],
            gap=0.6,
        )
    else:
        mcr_result = None
        mcr_panel = mo.md("MCR-ALS disabled.").callout(kind="neutral")
    return (mcr_panel,)


@app.cell(hide_code=True)
def _(
    controls,
    heatmap_panel,
    mcr_component_control,
    mcr_panel,
    mo,
    preprocess_summary,
    spectrum_panel,
):
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([controls, mcr_component_control, preprocess_summary], gap=0.8), heatmap_panel, spectrum_panel],
                widths=[1.05, 1.8, 1.45],
                align="start",
                gap=1.2,
            ),
            mcr_panel,
            mo.md(
                """
                #### Composable API sketch

                ```python
                band = current_map.extract_map((1550, 1650), agg="mean")
                current_map.plot(overlay=band)

                corrected = current_map.post_process(lambda s: s - np.median(s), by="spectrum")
                components = ramanrs.mcr_als(corrected.spectra, n_components=3, axis=corrected.raman_shift, shape=corrected.shape)
                current_map.plot(overlay=components.heatmap(0))
                ```
                """
            ),
        ],
        gap=1,
    )
    return
