"""
ramanrs — a Python reader for .rs Raman microscopy map files.

The .rs container is a tagged binary tree (big-endian) of named values.
Leaf "Serialised…" fields hold .NET BinaryFormatter streams (little-endian),
which in these files are plain one-dimensional primitive arrays. The optical
image is an embedded PNG.

Quick start
-----------
    import ramanrs

    ds = ramanrs.read("Map_LZ-18N_100x.rs")
    print(ds.title)

    m = ds.maps[0]                 # first RamanMap
    m.save_image("optical.png")    # composite optical image (PNG)

    counts = m.spectra             # (n_points, n_pixels) detector counts
    shift  = m.raman_shift         # Raman shift axis in cm^-1 (per pixel)
    x, y   = m.x, m.y              # stage coordinates of each point (um)
    grid   = m.spectra_grid        # (ny, nx, n_pixels) on the map grid

    # Extract a scalar Raman heatmap, then overlay it on the optical image
    band = m.extract_map((1550, 1650), axis="raman_shift", agg="mean")
    fig, ax = m.plot(overlay=band, alpha=0.65)
    fig.savefig("overlay.png")

    # Per-spectrum clean-up composes with component analysis
    cleaned = m.post_process(ramanrs.remove_cosmics, by="spectrum")
    components = cleaned.mcr_als(n_components=3)
    fig, ax = m.plot(overlay=components.heatmap(0))

NumPy is used for numeric arrays; plotting (``plot``, ``optical_image``)
uses Matplotlib.

Post-processing functions stored in the file (cropping, background
removal, …) are exposed via ``RamanMap.functions`` but are NOT applied;
only the two calibration steps (pixel -> wavelength -> Raman shift) are
used, and only when you ask for the corresponding axis.

Low-level access: ``ramanrs.parse(path)`` returns the raw tree as nested
dicts/lists, with "Serialised…" blobs left as bytes; ``ds.raw`` on a
Dataset gives the same tree. ``decode_dotnet_array(blob)`` decodes a
.NET BinaryFormatter primitive array.
"""

import io
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Iterator, Optional, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

__all__ = [
    "read",
    "parse",
    "decode_dotnet_array",
    "Dataset",
    "RamanMap",
    "RamanHeatmap",
    "RamanComponents",
    "remove_cosmics",
    "mcr_als",
    "Point",
    "RSFormatError",
]

__version__ = "0.1.0"


class RSFormatError(ValueError):
    """Raised when the file does not match the expected .rs structure."""


# ---------------------------------------------------------------------------
# Low-level container parser
#
# The container is a sequence of named, typed values:
#
#     [type: u8] [name length: u16 BE] [name: UTF-8] [payload]
#
# Payloads by type code (all multi-byte integers/floats big-endian):
#     0x01  boolean      1 byte
#     0x02  int16        2 bytes                  (unobserved; inferred)
#     0x03  int32        4 bytes
#     0x04  int64        8 bytes                  (unobserved; inferred)
#     0x05  float32      4 bytes                  (unobserved; inferred)
#     0x06  float64      8 bytes
#     0x07  binary blob  u32 BE length + bytes
#     0x08  string       u16 BE length + UTF-8 bytes
#     0x09  list         element type u8, count u32 BE, then payloads
#                        (no per-element names; dict elements are bare
#                        name/value sequences terminated by 0x00)
#     0x0A  dict         name/value entries terminated by a 0x00 byte
#
# The file as a whole is a single named dict (typically "Measurements").
# ---------------------------------------------------------------------------


class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def u8(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def take(self, n: int) -> bytes:
        b = self.data[self.pos : self.pos + n]
        if len(b) != n:
            raise RSFormatError(
                f"unexpected end of file at offset {self.pos} (wanted {n} bytes)"
            )
        self.pos += n
        return b

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        (v,) = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return v

    def name(self) -> str:
        n = self.unpack(">H")
        return self.take(n).decode("utf-8")

    def value(self, t: int) -> Any:
        if t == 0x01:
            return self.u8() != 0
        if t == 0x02:
            return self.unpack(">h")
        if t == 0x03:
            return self.unpack(">i")
        if t == 0x04:
            return self.unpack(">q")
        if t == 0x05:
            return self.unpack(">f")
        if t == 0x06:
            return self.unpack(">d")
        if t == 0x07:
            n = self.unpack(">I")
            return self.take(n)
        if t == 0x08:
            n = self.unpack(">H")
            return self.take(n).decode("utf-8")
        if t == 0x09:
            elem_t = self.u8()
            count = self.unpack(">I")
            return [self.value(elem_t) for _ in range(count)]
        if t == 0x0A:
            return self.mapping()
        raise RSFormatError(
            f"unknown value type 0x{t:02x} at offset {self.pos - 1}"
        )

    def mapping(self) -> dict:
        d: dict[str, Any] = {}
        while True:
            t = self.u8()
            if t == 0x00:
                return d
            name = self.name()  # name precedes the value in the stream
            d[name] = self.value(t)


def parse(source: Union[str, bytes, BinaryIO]) -> dict:
    """Parse an .rs file into its raw tree of nested dicts and lists.

    ``source`` may be a filename, a bytes object, or a binary file object.
    "Serialised…" fields and the optical image are returned as raw bytes.
    The returned dict has a single entry for the root node (usually
    ``"Measurements"``).
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
    else:
        with open(source, "rb") as fh:
            data = fh.read()

    r = _Reader(data)
    t = r.u8()
    if t != 0x0A:
        raise RSFormatError(
            f"expected a dict (0x0a) at the root, found 0x{t:02x};"
            " this does not look like an .rs measurement file"
        )
    root_name = r.name()
    tree = {root_name: r.mapping()}
    if r.pos != len(data):
        raise RSFormatError(
            f"trailing data after root node ({len(data) - r.pos} bytes)"
        )
    return tree


# ---------------------------------------------------------------------------
# .NET BinaryFormatter primitive arrays
#
# The "Serialised…" blobs are BinaryFormatter streams containing a single
# ArraySinglePrimitive record:
#
#     SerializationHeaderRecord  (17 bytes: 0x00 + four int32)
#     ArraySinglePrimitive       0x0F, objectId i32, length i32,
#                                PrimitiveTypeEnum u8, packed values (LE)
#     MessageEnd                 0x0B
# ---------------------------------------------------------------------------

_DOTNET_PRIMITIVES = {
    1: ("?", "boolean"),
    2: ("B", "byte"),
    6: ("d", "float64"),
    7: ("h", "int16"),
    8: ("i", "int32"),
    9: ("q", "int64"),
    10: ("b", "sbyte"),
    11: ("f", "float32"),
    14: ("H", "uint16"),
    15: ("I", "uint32"),
    16: ("Q", "uint64"),
}


def decode_dotnet_array(blob: bytes):
    """Decode a .NET BinaryFormatter stream holding one primitive array.

    Returns a NumPy ndarray. Raises :class:`RSFormatError` for anything
    other than a single ArraySinglePrimitive record.
    """
    if len(blob) < 19 or blob[0] != 0x00:
        raise RSFormatError("not a .NET BinaryFormatter stream")
    pos = 17  # skip SerializationHeaderRecord
    record = blob[pos]
    if record != 0x0F:
        raise RSFormatError(
            f"unsupported BinaryFormatter record 0x{record:02x}"
            " (only ArraySinglePrimitive is supported)"
        )
    length = struct.unpack_from("<i", blob, pos + 5)[0]
    prim = blob[pos + 9]
    try:
        code, _ = _DOTNET_PRIMITIVES[prim]
    except KeyError:
        raise RSFormatError(f"unsupported .NET primitive type {prim}") from None
    start = pos + 10
    nbytes = length * struct.calcsize(code)
    payload = blob[start : start + nbytes]
    if len(payload) != nbytes:
        raise RSFormatError("truncated BinaryFormatter array")
    return np.frombuffer(payload, dtype="<" + code).copy()


# ---------------------------------------------------------------------------
# High-level objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RamanHeatmap:
    """A scalar image derived from a :class:`RamanMap`.

    ``values`` is a two-dimensional ``(ny, nx)`` array aligned with the map
    grid. Use :meth:`RamanMap.extract_map` to make these from spectral bands
    or from a custom spectrum-to-scalar function, then pass the result to
    ``RamanMap.plot(overlay=...)``.
    """

    values: np.ndarray
    label: str = "Raman intensity"
    unit: str = "counts"
    cmap: str = "inferno"
    source: Optional["RamanMap"] = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("RamanHeatmap values must be a 2D array")
        object.__setattr__(self, "values", values)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def plot(self, *, ax=None, colorbar: bool = True, cmap: Optional[str] = None):
        """Plot the heatmap by itself. Returns ``(figure, axes)``."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure
        image = ax.imshow(self.values, origin="upper", cmap=cmap or self.cmap)
        ax.set(title=self.label)
        if colorbar:
            cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(self.label if not self.unit else f"{self.label} ({self.unit})")
        return fig, ax

    def __repr__(self) -> str:
        return f"<RamanHeatmap {self.label!r}: {self.shape[1]}x{self.shape[0]}>"


@dataclass(frozen=True)
class RamanComponents:
    """Spectral components and their spatial loadings.

    ``loadings`` is ``(n_points, n_components)`` and ``spectra`` is
    ``(n_components, n_pixels)``. Use :meth:`heatmap` to turn a component's
    spatial loading into a :class:`RamanHeatmap`.
    """

    loadings: np.ndarray
    spectra: np.ndarray
    axis: Optional[np.ndarray] = None
    shape: Optional[tuple[int, int]] = None
    labels: Optional[tuple[str, ...]] = None
    method: str = "MCR-ALS"
    reconstruction_error: Optional[float] = None
    source: Optional["RamanMap"] = None

    def __post_init__(self) -> None:
        loadings = np.asarray(self.loadings, dtype=float)
        spectra = np.asarray(self.spectra, dtype=float)
        if loadings.ndim != 2 or spectra.ndim != 2:
            raise ValueError("loadings and spectra must both be 2D arrays")
        if loadings.shape[1] != spectra.shape[0]:
            raise ValueError("loadings and spectra disagree on component count")
        if self.axis is not None and len(self.axis) != spectra.shape[1]:
            raise ValueError("axis length must match component spectra length")
        if self.shape is not None and np.prod(self.shape) != loadings.shape[0]:
            raise ValueError("shape must contain one pixel per loading row")
        if self.labels is None:
            labels = tuple(f"component {i + 1}" for i in range(spectra.shape[0]))
        else:
            labels = tuple(self.labels)
            if len(labels) != spectra.shape[0]:
                raise ValueError("labels length must match component count")

        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "axis", None if self.axis is None else np.asarray(self.axis, dtype=float))
        object.__setattr__(self, "labels", labels)

    @property
    def n_components(self) -> int:
        return self.spectra.shape[0]

    def heatmap(
        self,
        component: int,
        *,
        label: Optional[str] = None,
        cmap: str = "viridis",
    ) -> RamanHeatmap:
        """Spatial loading map for a component, using zero-based indexing."""
        if self.shape is None:
            raise ValueError("component heatmaps require a map shape")
        index = int(component)
        if not (0 <= index < self.n_components):
            raise IndexError(f"component must be in 0..{self.n_components - 1}")
        return RamanHeatmap(
            self.loadings[:, index].reshape(self.shape),
            label=label or f"{self.method} {self.labels[index]}",
            unit="a.u.",
            cmap=cmap,
            source=self.source,
        )

    def plot_spectra(self, *, ax=None):
        """Plot component spectra. Returns ``(figure, axes)``."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        x = np.arange(self.spectra.shape[1]) if self.axis is None else self.axis
        for i, spectrum in enumerate(self.spectra):
            ax.plot(x, spectrum, label=self.labels[i])
        ax.set(
            xlabel="Raman shift (cm$^{-1}$)" if self.axis is not None else "Pixel",
            ylabel="Component spectrum (normalised)",
            title=self.method,
        )
        ax.legend(frameon=False)
        return fig, ax

    def __repr__(self) -> str:
        err = "" if self.reconstruction_error is None else f", error={self.reconstruction_error:.3g}"
        return (
            f"<RamanComponents {self.method}: {self.n_components} components, "
            f"{self.loadings.shape[0]} points x {self.spectra.shape[1]} pixels{err}>"
        )


def remove_cosmics(
    spectra,
    *,
    window: int = 7,
    threshold: float = 8.0,
    spatial_threshold: Optional[float] = 12.0,
    wide_window: Optional[int] = None,
    iterations: int = 2,
    chunk_size: int = 512,
    return_mask: bool = False,
):
    """Remove narrow positive cosmic-ray spikes from spectra.

    A local median estimates the non-spike signal; points whose positive
    residual is large relative to a per-spectrum robust noise estimate are
    replaced by that local median. For spectra matrices, a second spatial
    outlier check catches multi-pixel spikes that are narrow spectrally but
    too wide to be found by the small local median window. Accepts either one
    spectrum ``(n_pixels,)`` or a spectra matrix ``(n_spectra, n_pixels)``.
    """
    data = np.asarray(spectra, dtype=float)
    if data.ndim == 1:
        cleaned = data[None, :].copy()
        single_spectrum = True
    elif data.ndim == 2:
        cleaned = data.copy()
        single_spectrum = False
    else:
        raise ValueError("spectra must be 1D or 2D")

    window = int(window)
    if window < 3:
        raise ValueError("window must be at least 3")
    if window % 2 == 0:
        window += 1
    radius = window // 2

    wide_window = max(window * 3, 21) if wide_window is None else int(wide_window)
    if wide_window % 2 == 0:
        wide_window += 1

    def row_median(values: np.ndarray, width: int) -> np.ndarray:
        radius = width // 2
        out = np.empty_like(values)
        for start in range(0, values.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), values.shape[0])
            chunk = values[start:stop]
            padded = np.pad(chunk, ((0, 0), (radius, radius)), mode="reflect")
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=width, axis=1
            )
            out[start:stop] = np.median(windows, axis=-1)
        return out

    def row_noise(residual: np.ndarray, values: np.ndarray) -> np.ndarray:
        centre = np.median(residual, axis=1, keepdims=True)
        mad = np.median(np.abs(residual - centre), axis=1, keepdims=True)
        diff_scale = np.median(np.abs(np.diff(values, axis=1)), axis=1, keepdims=True)
        noise = np.maximum(1.4826 * mad, diff_scale / np.sqrt(2))
        return np.maximum(noise, 1e-12)

    mask = np.zeros(cleaned.shape, dtype=bool)
    for _ in range(int(iterations)):
        iteration_mask = np.zeros(cleaned.shape, dtype=bool)

        if not single_spectrum and spatial_threshold is not None:
            wide_local = row_median(cleaned, wide_window)
            wide_residual = cleaned - wide_local
            spectral_noise = row_noise(wide_residual, cleaned)
            spatial_centre = np.median(cleaned, axis=0, keepdims=True)
            spatial_mad = np.median(np.abs(cleaned - spatial_centre), axis=0, keepdims=True)
            spatial_noise = np.maximum(1.4826 * spatial_mad, 1e-12)
            spatial_residual = cleaned - spatial_centre
            spatial_spikes = (
                (spatial_residual > float(spatial_threshold) * spatial_noise)
                & (wide_residual > float(threshold) * spectral_noise)
            )
            replacement = np.minimum(wide_local, spatial_centre)
            cleaned[spatial_spikes] = replacement[spatial_spikes]
            iteration_mask |= spatial_spikes

        for start in range(0, cleaned.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), cleaned.shape[0])
            chunk = cleaned[start:stop]
            padded = np.pad(chunk, ((0, 0), (radius, radius)), mode="reflect")
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=window, axis=1
            )
            local = np.median(windows, axis=-1)
            residual = chunk - local
            noise = row_noise(residual, chunk)

            spikes = residual > float(threshold) * noise
            chunk[spikes] = local[spikes]
            iteration_mask[start:stop] |= spikes

        mask |= iteration_mask
        if not iteration_mask.any():
            break

    cleaned = cleaned[0] if single_spectrum else cleaned
    mask = mask[0] if single_spectrum else mask
    return (cleaned, mask) if return_mask else cleaned


def _pure_spectra_initial_guess(data: np.ndarray, n_components: int) -> np.ndarray:
    """Successive orthogonal-projection initial spectra for MCR-ALS."""
    eps = 1e-12
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalised = data / np.maximum(norms, eps)
    residual = normalised.copy()
    chosen: list[int] = []

    for _ in range(n_components):
        scores = np.sum(residual * residual, axis=1)
        for index in np.argsort(scores)[::-1]:
            index = int(index)
            if index not in chosen:
                chosen.append(index)
                break
        basis, _ = np.linalg.qr(normalised[chosen].T)
        residual = normalised - normalised @ basis @ basis.T

    return data[chosen].copy()


def mcr_als(
    spectra,
    n_components: int = 3,
    *,
    n_iter: int = 80,
    init: str = "pure",
    spectral_sparsity: float = 0.01,
    closure: bool = False,
    random_state: int = 0,
    axis=None,
    shape: Optional[tuple[int, int]] = None,
    labels: Optional[tuple[str, ...]] = None,
    source: Optional["RamanMap"] = None,
) -> RamanComponents:
    """Resolve mixture spectra with MCR-ALS.

    MCR-ALS alternates between non-negative least-squares estimates of spatial
    loadings and component spectra. ``spectral_sparsity`` is a small soft
    threshold on component spectra; it helps reduce the rotational ambiguity
    that otherwise lets the same bands appear in multiple components.
    """
    data = np.asarray(spectra, dtype=float)
    if data.ndim != 2:
        raise ValueError("spectra must be a 2D (n_points, n_pixels) array")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0)

    n_points, n_pixels = data.shape
    n_components = int(n_components)
    if not (1 <= n_components <= min(n_points, n_pixels)):
        raise ValueError("n_components must fit inside the spectra matrix")

    eps = 1e-12
    if init == "pure":
        component_spectra = _pure_spectra_initial_guess(data, n_components)
    elif init == "svd":
        _, _, vt = np.linalg.svd(data, full_matrices=False)
        component_spectra = np.maximum(vt[:n_components], 0.0)
        empty = component_spectra.max(axis=1) <= eps
        component_spectra[empty] = np.abs(vt[:n_components][empty])
    elif init == "random":
        rng = np.random.default_rng(random_state)
        component_spectra = rng.random((n_components, n_pixels))
    else:
        component_spectra = np.asarray(init, dtype=float)
        if component_spectra.shape != (n_components, n_pixels):
            raise ValueError(
                "custom init must have shape (n_components, n_pixels)"
            )
        component_spectra = np.maximum(component_spectra, 0.0)

    loadings = np.zeros((n_points, n_components), dtype=float)
    for _ in range(int(n_iter)):
        loadings = np.linalg.lstsq(component_spectra.T, data.T, rcond=None)[0].T
        loadings = np.maximum(loadings, 0.0)
        if closure:
            row_sum = np.maximum(loadings.sum(axis=1, keepdims=True), eps)
            loadings = loadings / row_sum

        component_spectra = np.linalg.lstsq(loadings, data, rcond=None)[0]
        component_spectra = np.maximum(component_spectra, 0.0)
        if spectral_sparsity:
            threshold = float(spectral_sparsity) * np.maximum(
                component_spectra.max(axis=1, keepdims=True), eps
            )
            component_spectra = np.maximum(component_spectra - threshold, 0.0)

        scale = np.maximum(component_spectra.max(axis=1), eps)
        component_spectra = component_spectra / scale[:, None]
        loadings = loadings * scale[None, :]

    order = np.argsort(loadings.sum(axis=0))[::-1]
    loadings = loadings[:, order]
    component_spectra = component_spectra[order]
    if labels is not None:
        labels = tuple(labels[i] for i in order)

    reconstructed = loadings @ component_spectra
    error = float(np.linalg.norm(data - reconstructed) / np.maximum(np.linalg.norm(data), eps))
    return RamanComponents(
        loadings=loadings,
        spectra=component_spectra,
        axis=axis,
        shape=shape,
        labels=labels,
        method="MCR-ALS",
        reconstruction_error=error,
        source=source,
    )


class Point:
    """A single measured point of a map: one spectrum plus its metadata."""

    def __init__(self, node: dict, parent: "RamanMap" = None):
        self._node = node
        self._parent = parent  # for spectral axes, which live on the map

    @property
    def raw(self) -> dict:
        """Raw tree node for this point."""
        return self._node

    @property
    def label(self) -> str:
        return self._node.get("Label", "")

    @property
    def metadata(self) -> dict:
        """Per-point metadata (typically X, Y, Z stage coordinates in um)."""
        return self._node.get("Metadata", {})

    @property
    def position(self) -> tuple:
        """(x, y, z) stage coordinates in micrometres, where present."""
        md = self.metadata
        return (md.get("X"), md.get("Y"), md.get("Z"))

    @property
    def n_pixels(self) -> Optional[int]:
        axes = self._node.get("DomainAxes") or []
        return axes[0].get("NumberOfPoints") if axes else None

    @property
    def counts(self):
        """Detector counts for this spectrum, one value per camera pixel."""
        return decode_dotnet_array(self._node["SerialisedData"])

    @property
    def raman_shift(self):
        """Raman shift axis in cm^-1 (shared map calibration), or None."""
        return None if self._parent is None else self._parent.raman_shift

    @property
    def wavelength(self):
        """Wavelength axis in nm (shared map calibration), or None."""
        return None if self._parent is None else self._parent.wavelength

    def plot(self, x: str = "raman_shift", ax=None, **kwargs):
        """Plot this spectrum. ``x`` is "raman_shift", "wavelength" or "pixel".

        Extra keyword arguments are passed to ``Axes.plot``. Returns
        (figure, axes). Requires Matplotlib.
        """
        labels = {
            "raman_shift": "Raman shift (cm$^{-1}$)",
            "wavelength": "Wavelength (nm)",
            "pixel": "Camera pixel",
        }
        if x not in labels:
            raise ValueError(f"x must be one of {sorted(labels)}, not {x!r}")
        counts = self.counts
        if x == "pixel":
            xs = range(len(counts))
        else:
            xs = getattr(self, x)
            if xs is None:
                raise RSFormatError(f"no {x} calibration available")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        kwargs.setdefault("linewidth", 1)
        ax.plot(xs, counts, **kwargs)
        ax.set(xlabel=labels[x], ylabel="Counts", title=self.label)
        return fig, ax

    def __repr__(self) -> str:
        return f"<Point {self.label!r}, {self.n_pixels} pixels>"


class RamanMap:
    """A Raman map: an optical image plus spectra on a grid of points."""

    def __init__(self, node: dict):
        self._node = node
        self._points = [Point(p, self) for p in node.get("Points", [])]

    # -- point lookup -----------------------------------------------------------

    def point_at(self, row: int, col: int) -> Point:
        """The point at grid position (row, col); row 0 has the smallest Y."""
        shape = self.shape
        if shape is None:
            raise RSFormatError("map grid dimensions not present in metadata")
        ny, nx = shape
        if not (0 <= row < ny and 0 <= col < nx):
            raise IndexError(f"(row, col) out of range for a {ny}x{nx} grid")
        return self._points[row * nx + col]

    def nearest_point(self, x: float, y: float, coords: str = "relative") -> Point:
        """The point nearest (x, y) in um.

        ``coords`` is "relative" (from the optical image's top-left corner,
        matching the default ``plot`` axes) or "stage" (absolute).
        """
        if coords == "relative":
            ext = self.image_extent
            if ext is None:
                raise RSFormatError("image geometry missing from file")
            x, y = x + ext[0], y + ext[2]
        elif coords != "stage":
            raise ValueError(f"coords must be 'relative' or 'stage', not {coords!r}")
        xs, ys = self.x, self.y
        best = min(
            range(len(self._points)),
            key=lambda i: (xs[i] - x) ** 2 + (ys[i] - y) ** 2,
        )
        return self._points[best]

    # -- general -----------------------------------------------------------

    @property
    def raw(self) -> dict:
        """Raw tree node for this map."""
        return self._node

    @property
    def label(self) -> str:
        return self._node.get("Label", "")

    @property
    def metadata(self) -> dict:
        """Acquisition metadata (laser, grating, exposure, grid layout, …)."""
        return self._node.get("Metadata", {})

    @property
    def functions(self) -> list:
        """Calibration / post-processing function definitions, as stored.

        These are returned verbatim (raw dicts) and are not applied to the
        data, except that :attr:`wavelength` and :attr:`raman_shift` use the
        two calibration entries (``PolynomialFitCalibrationFunction`` and
        ``LaserLineCalibrationFunction``) to construct spectral axes.
        """
        return self._node.get("MapFunctions", [])

    def _function(self, type_name: str) -> Optional[dict]:
        for f in self.functions:
            if f.get("Type") == type_name:
                return f
        return None

    # -- optical image ------------------------------------------------------

    @property
    def image_png(self) -> Optional[bytes]:
        """The composite optical image as PNG bytes (or None if absent)."""
        comp = self._node.get("CompositeImage") or {}
        return comp.get("Image")

    @property
    def image_size(self) -> Optional[tuple]:
        """(width, height) of the optical image in pixels."""
        comp = self._node.get("CompositeImage") or {}
        w, h = comp.get("ImagePixelWidth"), comp.get("ImagePixelHeight")
        return (w, h) if w is not None else None

    def save_image(self, path: str) -> None:
        """Write the optical image to ``path`` as a PNG file."""
        png = self.image_png
        if png is None:
            raise RSFormatError("this map has no composite image")
        with open(path, "wb") as fh:
            fh.write(png)

    # -- points and spectra --------------------------------------------------

    @property
    def points(self) -> list:
        """The measured points, in acquisition (storage) order."""
        return self._points

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[Point]:
        return iter(self._points)

    def __getitem__(self, i) -> Point:
        return self._points[i]

    @property
    def n_pixels(self) -> Optional[int]:
        """Number of camera pixels per spectrum."""
        return self._points[0].n_pixels if self._points else None

    @property
    def spectra(self):
        """All spectra as a (n_points, n_pixels) NumPy array of counts."""
        return np.vstack([p.counts for p in self._points])

    def _axis_values(self, label: str):
        for ax in self._node.get("DomainAxes", []):
            if ax.get("Label") == label and "SerialisedPoints" in ax:
                return decode_dotnet_array(ax["SerialisedPoints"])
        return None

    @property
    def x(self):
        """Per-point X stage coordinates (um), in storage order."""
        return self._axis_values("X")

    @property
    def y(self):
        """Per-point Y stage coordinates (um), in storage order."""
        return self._axis_values("Y")

    @property
    def z(self):
        """Per-point Z stage coordinates (um), in storage order."""
        return self._axis_values("Z")

    # -- grid layout ----------------------------------------------------------

    @property
    def shape(self) -> Optional[tuple]:
        """Map grid shape (ny, nx) from the acquisition metadata."""
        md = self.metadata
        nx, ny = md.get("Map X Count"), md.get("Map Y Count")
        if nx is None or ny is None:
            return None
        return (int(ny), int(nx))

    @property
    def spectra_grid(self):
        """Spectra arranged on the map grid as (ny, nx, n_pixels).

        Points are stored row by row with X varying fastest, which matches
        the stage coordinates recorded per point.
        """
        shape = self.shape
        if shape is None:
            raise RSFormatError("map grid dimensions not present in metadata")
        ny, nx = shape
        if ny * nx != len(self._points):
            raise RSFormatError(
                f"grid {ny}x{nx} does not match {len(self._points)} points"
            )
        return self.spectra.reshape(ny, nx, -1)

    # -- spectral axes ----------------------------------------------------------

    @property
    def pixel(self):
        """Camera pixel indices 0..n_pixels-1 (the stored domain axis)."""
        n = self.n_pixels
        if n is None:
            return None
        return np.arange(n)

    @property
    def wavelength_coefficients(self):
        """Polynomial coefficients c such that lambda(p) = sum c[k] * p**k."""
        f = self._function("PolynomialFitCalibrationFunction")
        if f is None or "SerialisedCoefficients" not in f:
            return None
        return decode_dotnet_array(f["SerialisedCoefficients"])

    @property
    def wavelength(self):
        """Wavelength axis in nm, from the stored polynomial calibration."""
        coeffs = self.wavelength_coefficients
        n = self.n_pixels
        if coeffs is None or n is None:
            return None
        p = np.arange(n, dtype=float)
        return np.polynomial.polynomial.polyval(p, np.asarray(coeffs))

    @property
    def laser_wavelength(self) -> Optional[float]:
        """Nominal laser wavelength in nm, if recorded."""
        f = self._function("LaserLineCalibrationFunction")
        return None if f is None else f.get("Laser")

    @property
    def raman_shift(self):
        """Raman shift axis in cm^-1: offset - 1e7 / wavelength(nm).

        The offset is the calibrated laser-line wavenumber stored in the
        file's LaserLineCalibrationFunction. Note the axis decreases with
        pixel index on this instrument; sort or flip as needed.
        """
        f = self._function("LaserLineCalibrationFunction")
        wl = self.wavelength
        if f is None or wl is None:
            return None
        offset = f.get("Offset")
        if offset is None:
            laser = f.get("Laser")
            if not laser:
                return None
            offset = 1e7 / laser
        return offset - 1e7 / wl

    # -- physical geometry ----------------------------------------------------
    #
    # The composite image's SerialisedDimensions blob holds
    # (x_origin, y_origin, z_origin, width, height, depth) in micrometres,
    # in stage coordinates. Image row 0 corresponds to y = y_origin, with
    # stage Y increasing down the image (screen convention), as verified
    # against the Raman signal of a reference map.

    @property
    def image_dimensions(self) -> Optional[tuple]:
        """(x0, y0, z0, width, height, depth) of the optical image in um."""
        comp = self._node.get("CompositeImage") or {}
        blob = comp.get("SerialisedDimensions")
        if blob is None:
            return None
        return tuple(decode_dotnet_array(blob))

    @property
    def image_extent(self) -> Optional[tuple]:
        """Optical image extent (x_min, x_max, y_min, y_max) in stage um."""
        dims = self.image_dimensions
        if dims is None:
            return None
        x0, y0, _, w, h, _ = dims
        return (x0, x0 + w, y0, y0 + h)

    @property
    def point_separation(self) -> Optional[tuple]:
        """Grid point spacing (dx, dy) in um, from the acquisition metadata."""
        md = self.metadata
        dx, dy = md.get("Map X Separation"), md.get("Map Y Separation")
        if dx is None or dy is None:
            return None
        return (float(dx), float(dy))

    @property
    def grid_extent(self) -> Optional[tuple]:
        """Heatmap cell-edge extent (x_min, x_max, y_min, y_max) in stage um.

        Each grid point owns a cell of one point separation centred on it,
        so the extent reaches half a separation beyond the outermost points.
        """
        x, y, sep = self.x, self.y, self.point_separation
        if x is None or y is None or sep is None:
            return None
        dx, dy = sep
        return (
            min(x) - dx / 2, max(x) + dx / 2,
            min(y) - dy / 2, max(y) + dy / 2,
        )

    # -- derived maps and post-processing -------------------------------------

    def extract_map(
        self,
        source,
        *,
        axis: str = "raman_shift",
        agg: Union[str, Callable] = "mean",
        label: Optional[str] = None,
        unit: str = "counts",
        cmap: str = "inferno",
    ) -> RamanHeatmap:
        """Extract a scalar :class:`RamanHeatmap` from this map.

        ``source`` can be either:

        - ``(low, high)``: a spectral window on ``axis`` (``"raman_shift"``,
          ``"wavelength"`` or ``"pixel"``), aggregated with ``agg``.
        - ``callable``: a function called once per spectrum; it must return a
          scalar. Use closures if the function needs the spectral axis.
        - a 2D array already shaped like ``map.shape``.
        """
        shape = self.shape
        if shape is None:
            raise RSFormatError("map grid dimensions not present in metadata")

        if isinstance(source, RamanHeatmap):
            if source.shape != shape:
                raise ValueError(
                    f"heatmap shape {source.shape} does not match map shape {shape}"
                )
            return source

        if callable(source):
            values = np.asarray([source(s.copy()) for s in self.spectra], dtype=float)
            if values.shape != (len(self),):
                raise ValueError(
                    "a spectrum-to-map function must return one scalar per spectrum"
                )
            return RamanHeatmap(
                values.reshape(shape),
                label=label or getattr(source, "__name__", "derived map"),
                unit=unit,
                cmap=cmap,
                source=self,
            )

        arr = np.asarray(source)
        if arr.shape == shape:
            return RamanHeatmap(
                arr, label=label or "Raman map", unit=unit, cmap=cmap, source=self
            )

        try:
            lo, hi = sorted(float(v) for v in source)
        except (TypeError, ValueError):
            raise ValueError(
                "source must be a (low, high) window, a spectrum-to-scalar "
                "function, or a 2D array shaped like map.shape"
            ) from None

        axis_values, axis_label, axis_unit = self._spectral_axis(axis)
        mask = (axis_values >= lo) & (axis_values <= hi)
        if not mask.any():
            raise ValueError(
                f"window {lo:g}..{hi:g} {axis_unit} is outside the {axis} axis "
                f"({float(axis_values.min()):.1f}..{float(axis_values.max()):.1f})"
            )

        selected = self.spectra[:, mask].astype(float)
        if callable(agg):
            values = np.asarray([agg(s.copy()) for s in selected], dtype=float)
            agg_name = getattr(agg, "__name__", "custom")
        else:
            reducers = {
                "mean": np.mean,
                "sum": np.sum,
                "max": np.max,
                "min": np.min,
                "median": np.median,
            }
            try:
                reducer = reducers[agg]
            except KeyError:
                names = ", ".join(sorted(reducers))
                raise ValueError(f"unknown agg {agg!r}; use one of {names}") from None
            values = reducer(selected, axis=1)
            agg_name = agg

        if values.shape != (len(self),):
            raise ValueError("agg must produce one scalar per spectrum")
        default_label = f"{agg_name} {lo:g}\u2013{hi:g} {axis_label}"
        return RamanHeatmap(
            values.reshape(shape),
            label=label or default_label,
            unit=unit,
            cmap=cmap,
            source=self,
        )

    def heatmap(self, window: tuple, agg: str = "mean"):
        """Band intensity on the map grid for a Raman-shift window.

        This is the array-only convenience form of ``extract_map(window)``.
        Prefer ``extract_map`` when the result will be plotted or composed.
        """
        return self.extract_map(window, axis="raman_shift", agg=agg).values

    def post_process(
        self,
        function: Callable,
        *,
        by: str = "map",
        label: Optional[str] = None,
    ):
        """Apply ``function`` without mutating this map.

        ``by="map"`` passes the full ``(ny, nx, n_pixels)`` spectra cube to
        ``function``. Return a cube of the same shape for a new RamanMap-like
        view, or a ``(ny, nx)`` array for a :class:`RamanHeatmap`.

        ``by="spectrum"`` applies ``function`` to each 1D spectrum. Return a
        processed 1D spectrum to get a RamanMap-like view, or a scalar to get
        a :class:`RamanHeatmap`.

        Examples
        --------
            corrected = m.post_process(lambda s: s - np.median(s), by="spectrum")
            max_map = m.post_process(lambda cube: cube.max(axis=-1), by="map")
        """
        if by == "map":
            result = function(self.spectra_grid.copy())
        elif by == "spectrum":
            result = np.asarray([function(s.copy()) for s in self.spectra])
        else:
            raise ValueError(f"by must be 'map' or 'spectrum', not {by!r}")

        if isinstance(result, RamanHeatmap):
            return result

        values = np.asarray(result)
        name = label or "processed"
        if values.shape == self.spectra_grid.shape:
            return _ProcessedRamanMap(
                self, values.reshape(len(self), -1), label=name
            )
        if values.shape == self.spectra.shape:
            return _ProcessedRamanMap(self, values, label=name)
        if values.shape == self.shape:
            return RamanHeatmap(values, label=name, source=self)
        if values.shape == (len(self),):
            return RamanHeatmap(values.reshape(self.shape), label=name, source=self)

        raise ValueError(
            "post_process must return spectra shaped like the input, "
            "a 2D map, or one scalar per map point"
        )

    def mcr_als(self, n_components: int = 3, **kwargs) -> RamanComponents:
        """Resolve this map into component spectra and loading maps.

        Extra keyword arguments are passed to :func:`mcr_als`.
        """
        return mcr_als(
            self.spectra,
            n_components=n_components,
            axis=self.raman_shift,
            shape=self.shape,
            source=self,
            **kwargs,
        )

    def _spectral_axis(self, axis: str):
        axes = {
            "raman_shift": (self.raman_shift, "cm$^{-1}$", "cm^-1"),
            "wavelength": (self.wavelength, "nm", "nm"),
            "pixel": (self.pixel, "pixel", "pixel"),
        }
        try:
            values, label, unit = axes[axis]
        except KeyError:
            raise ValueError(
                f"axis must be one of {sorted(axes)}, not {axis!r}"
            ) from None
        if values is None:
            raise RSFormatError(f"no {axis} calibration available")
        return np.asarray(values, dtype=float), label, unit

    # -- optical image as pixels & plotting -----------------------------------

    def optical_image(self):
        """The optical image decoded to an (H, W, channels) array."""
        png = self.image_png
        if png is None:
            raise RSFormatError("this map has no composite image")
        return mpimg.imread(io.BytesIO(png), format="png")

    def plot(
        self,
        *,
        overlay: Optional[Union[RamanHeatmap, np.ndarray]] = None,
        alpha: float = 0.65,
        cmap: Optional[str] = None,
        interpolation: str = "nearest",
        coords: str = "relative",
        scalebar: Union[bool, float] = True,
        colorbar: bool = True,
        ax=None,
    ):
        """Plot the optical image, optionally with a ``RamanHeatmap`` overlay.

        Example
        -------
            band = raman_map.extract_map((1550, 1650))
            raman_map.plot(overlay=band)
        """
        img_ext = self.image_extent
        if img_ext is None:
            raise RSFormatError("image geometry missing from file")
        image = self.optical_image()

        heatmap = None
        grid_ext = None
        if overlay is not None:
            heatmap = self.extract_map(overlay) if not isinstance(overlay, RamanHeatmap) else overlay
            if heatmap.shape != self.shape:
                raise ValueError(
                    f"overlay shape {heatmap.shape} does not match map shape {self.shape}"
                )
            grid_ext = self.grid_extent
            if grid_ext is None:
                raise RSFormatError("grid geometry missing from file")

        ix0, ix1, iy0, iy1 = img_ext
        if grid_ext is not None:
            gx0, gx1, gy0, gy1 = grid_ext
        if coords == "relative":
            if grid_ext is not None:
                gx0, gx1, gy0, gy1 = gx0 - ix0, gx1 - ix0, gy0 - iy0, gy1 - iy0
            ix0, ix1, iy0, iy1 = 0.0, ix1 - ix0, 0.0, iy1 - iy0
        elif coords != "stage":
            raise ValueError(f"coords must be 'relative' or 'stage', not {coords!r}")

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(9, 9 * (iy1 - iy0) / (ix1 - ix0) + 0.5)
            )
        else:
            fig = ax.figure

        # extent=(left, right, bottom, top); image row 0 sits at y_min and
        # stage Y increases down the image, so the y axis points downward.
        ax.imshow(image, extent=(ix0, ix1, iy1, iy0), origin="upper")
        artist = None
        if heatmap is not None:
            artist = ax.imshow(
                heatmap.values,
                extent=(gx0, gx1, gy1, gy0),
                origin="upper",
                cmap=cmap or heatmap.cmap,
                alpha=alpha,
                interpolation=interpolation,
                zorder=2,
            )

        ax.set(
            xlim=(ix0, ix1),
            ylim=(iy1, iy0),
            xlabel="x (\u00b5m)",
            ylabel="y (\u00b5m)",
        )
        ax.set_aspect("equal")

        if colorbar and artist is not None:
            cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
            cb_label = heatmap.label if not heatmap.unit else f"{heatmap.label} ({heatmap.unit})"
            cb.set_label(cb_label)

        if scalebar:
            self._draw_scalebar(
                ax, None if scalebar is True else float(scalebar)
            )

        title = self.label or "Raman map"
        if heatmap is not None:
            title = f"{title} \u2014 {heatmap.label}"
        ax.set_title(title)
        return fig, ax

    @staticmethod
    def _draw_scalebar(ax, length_um: Optional[float]) -> None:
        """Draw a scale bar in the lower-left corner of ``ax`` (data in um)."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()  # y axis points downward: y0 > y1
        width = abs(x1 - x0)
        if length_um is None:  # nicest 1/2/5 x 10^k near a fifth of the width
            target = width / 5
            candidates = [
                m * 10**e for e in range(-2, 5) for m in (1, 2, 5)
            ]
            length_um = min(candidates, key=lambda c: abs(c - target))
        xpad = 0.04 * width
        ypad = 0.05 * abs(y1 - y0)
        bx = min(x0, x1) + xpad
        by = max(y0, y1) - ypad  # near the bottom (largest y)
        bh = 0.012 * abs(y1 - y0)
        ax.add_patch(
            Rectangle(
                (bx, by - bh), length_um, bh,
                facecolor="black", edgecolor="black",
                linewidth=0.6, zorder=5,
            )
        )
        label = (
            f"{length_um:g} \u00b5m"
            if length_um >= 1
            else f"{length_um * 1000:g} nm"
        )
        ax.text(
            bx + length_um / 2, by - bh - 0.012 * abs(y1 - y0), label,
            ha="center", va="bottom", color="black",
            fontsize=14, zorder=5,
            # path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
        )

    def __repr__(self) -> str:
        shape = self.shape
        grid = f"{shape[1]}x{shape[0]} grid, " if shape else ""
        return (
            f"<RamanMap {self.label!r}: {grid}{len(self)} points x "
            f"{self.n_pixels} pixels>"
        )


class _ProcessedPoint(Point):
    """Point view used by non-mutating RamanMap.post_process results."""

    def __init__(self, source: Point, parent: "_ProcessedRamanMap", counts):
        super().__init__(source.raw, parent)
        self._source = source
        self._counts = np.asarray(counts)

    @property
    def counts(self):
        """Processed detector counts for this spectrum."""
        return self._counts.copy()

    @property
    def n_pixels(self) -> int:
        return int(self._counts.shape[0])


class _ProcessedRamanMap(RamanMap):
    """RamanMap-like, non-mutating view over processed spectra."""

    def __init__(self, parent: RamanMap, spectra, label: str):
        spectra = np.asarray(spectra)
        if spectra.shape != parent.spectra.shape:
            raise ValueError(
                f"processed spectra must have shape {parent.spectra.shape}, "
                f"not {spectra.shape}"
            )
        self._parent_map = parent
        self._node = parent.raw
        self._spectra = spectra.copy()
        self._points = [
            _ProcessedPoint(point, self, counts)
            for point, counts in zip(parent.points, self._spectra)
        ]
        self._label = label

    @property
    def label(self) -> str:
        return self._label or self._parent_map.label

    @property
    def spectra(self):
        """All processed spectra as a (n_points, n_pixels) NumPy array."""
        return self._spectra.copy()

    @property
    def n_pixels(self) -> int:
        return int(self._spectra.shape[1])


class Dataset:
    """Top-level contents of an .rs file."""

    def __init__(self, tree: dict):
        if len(tree) != 1:
            raise RSFormatError("expected a single root node")
        (self.root_name,) = tree
        self._root = tree[self.root_name]
        self.maps = [
            RamanMap(m)
            for m in self._root.get("MeasurementList", [])
            if m.get("Type") == "RamanMap"
        ]
        self.measurements = self._root.get("MeasurementList", [])

    @property
    def raw(self) -> dict:
        """The full raw tree, including everything not exposed above."""
        return {self.root_name: self._root}

    @property
    def title(self) -> str:
        return self._root.get("Title", "")

    @property
    def label(self) -> str:
        return self._root.get("Label", "")

    @property
    def metadata(self) -> dict:
        return self._root.get("Metadata", {})

    def __repr__(self) -> str:
        return f"<Dataset {self.title!r}: {len(self.maps)} map(s)>"


def read(source: Union[str, bytes, BinaryIO]) -> Dataset:
    """Read an .rs file and return a :class:`Dataset`.

    ``source`` may be a filename, a bytes object, or a binary file object.
    """
    return Dataset(parse(source))


if __name__ == "__main__":
    import sys

    for filename in sys.argv[1:]:
        ds = read(filename)
        print(f"{filename}: {ds!r}")
        for k, v in ds.metadata.items():
            print(f"  {k}: {v}")
        for m in ds.maps:
            print(f"  {m!r}")
            for k, v in m.metadata.items():
                print(f"    {k}: {v}")
