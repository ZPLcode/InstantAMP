"""Motion Train's mild terrain distribution, represented by local heightfields.

The patch dimensions, quantization, cubic noise interpolation, pyramid slopes,
and reset origins follow ZENBOT-AMP.  The contact backend remains MuJoCo rather
than Motion Train's Newton external triangle collision pipeline.  No plane is
placed over the heightfield: negative roughness and inverted slopes stay real.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.interpolate import RectBivariateSpline


PATCH_SIZE = 8.0
REFERENCE_SPACING = .05
# Sample the source surface at every second point. At 10 cm even a rotated
# Zenbot23 collider spans at most 5x5 cells (50 triangle candidates) per pair.
COLLISION_STRIDE = 2
HORIZONTAL_SCALE = REFERENCE_SPACING * COLLISION_STRIDE
VERTICAL_SCALE = .0025
PATCH_BORDER = .50
ATLAS_BORDER = 10.0
PROPORTIONS = np.array([.40, .35 / 3, .35 / 3, .35 / 3, .05, .05, .05, .05, .05])
CORRELATION_LENGTHS = (.20, .30, .40, .15, .225, .30)


def _patch(kind, difficulty, rng):
    """Return a patch in [x, y] order, with the source's zero-height border."""
    side = round(PATCH_SIZE / REFERENCE_SPACING) + 1
    heights = np.zeros((side, side), np.float32)
    if kind == 0:
        return heights
    border = int(PATCH_BORDER / REFERENCE_SPACING) + 1
    inner = side - 2 * border
    width = inner * REFERENCE_SPACING
    if kind < 7:
        amplitude = 4 if kind <= 3 else 8  # in 2.5 mm height units
        step = 1 if kind <= 3 else 2
        samples = int(width / CORRELATION_LENGTHS[kind - 1])
        coarse = rng.choice(np.arange(-amplitude, amplitude + step, step), (samples, samples))
        # Rescaling both axes leaves the source's cubic interpolation unchanged.
        grid = np.linspace(0., 1., samples)
        interpolator = RectBivariateSpline(grid, grid, coarse)
        fine_grid = np.linspace(0., 1., inner)
        raw = interpolator(fine_grid, fine_grid)
    else:
        slope = (.02 + .04 * difficulty) * (1 if kind == 7 else -1)
        peak = int(slope * width / (2 * VERTICAL_SCALE))
        center = inner // 2
        ramp = 1 - np.abs(np.arange(inner) - center) / center
        raw = peak * ramp[:, None] * ramp[None, :]
        half_platform = int(2.0 / REFERENCE_SPACING / 2)
        edge = raw[center - half_platform, center - half_platform]
        raw = np.clip(raw, min(0., edge), max(0., edge))
    # Cubic interpolation can overshoot the sampled noise range, as in MT.
    heights[border:-border, border:-border] = np.rint(raw) * VERTICAL_SCALE
    return heights


@dataclass(frozen=True)
class TerrainAtlas:
    """Shared terrain atlas and matching differentiable height queries.

    ``heights`` is [y, x], matching MuJoCo's row-major heightfield convention.
    ``origins`` and ``kinds`` are [terrain row along x, terrain column along y].
    Each world samples its row once; columns are spread across the environments
    as in Motion Train's TerrainImporter, then shuffled so fixed walking and
    get-up environment blocks do not correlate with terrain columns.  The
    nominal category weights apply when generating patches, not as exact quotas
    in this finite atlas.  The physical atlas lies in positive XY; its leading
    10 m flat border keeps the reference animation origin at zero height.
    """

    heights: np.ndarray
    origins: np.ndarray
    kinds: np.ndarray
    half_size: np.ndarray

    @classmethod
    def create(cls, seed=0, *, rows=10, cols=20):
        if rows <= 0 or cols <= 0:
            raise ValueError("Terrain rows and columns must be positive")
        patch_cells = round(PATCH_SIZE / HORIZONTAL_SCALE)
        border_cells = round(ATLAS_BORDER / HORIZONTAL_SCALE)
        nx = rows * patch_cells + 2 * border_cells + 1
        ny = cols * patch_cells + 2 * border_cells + 1
        heights = np.zeros((ny, nx), np.float32)
        origins = np.zeros((rows, cols, 3), np.float32)
        kinds = np.zeros((rows, cols), np.int32)
        # The source uses a separate Generator for patch selection/difficulty.
        # Keeping noise generation separate preserves its seed-0 patch layout.
        layout_rng, noise_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        for row in range(rows):
            for col in range(cols):
                kind = layout_rng.choice(len(PROPORTIONS), p=PROPORTIONS)
                difficulty = layout_rng.uniform(0., 1.)
                patch = _patch(kind, difficulty, noise_rng)[::COLLISION_STRIDE, ::COLLISION_STRIDE]
                x0, y0 = border_cells + row * patch_cells, border_cells + col * patch_cells
                heights[y0:y0 + patch_cells + 1, x0:x0 + patch_cells + 1] = patch.T
                # MT uses the highest surface in the central 2 m square.
                lo, hi = round(3. / HORIZONTAL_SCALE), round(5. / HORIZONTAL_SCALE)
                origins[row, col] = ((row + .5 - rows / 2) * PATCH_SIZE,
                                     (col + .5 - cols / 2) * PATCH_SIZE,
                                     patch[lo:hi, lo:hi].max())
                kinds[row, col] = kind
        half_size = np.array([rows * PATCH_SIZE / 2 + ATLAS_BORDER,
                              cols * PATCH_SIZE / 2 + ATLAS_BORDER], np.float32)
        origins[..., :2] += half_size
        return cls(heights, origins, kinds, half_size)

    def add_to_spec(self, spec, floor):
        """Add one local collider per patch and four finite outer border boxes.

        The existing geom remains named ``floor``; all additional ground geoms
        are attached directly to world body 0.  Splitting the surface limits
        heightfield-local CCD coordinates to +/-4 m.  A single atlas-sized
        heightfield produced incorrect penetrations in float32 MuJoCo Warp.
        Flat patches use boxes so their contact uses the ordinary box pipeline.
        """
        cells = round(PATCH_SIZE / HORIZONTAL_SCALE)
        border = round(ATLAS_BORDER / HORIZONTAL_SCALE)
        material = dict(contype=floor.contype, conaffinity=floor.conaffinity,
                        friction=np.asarray(floor.friction).copy(), condim=floor.condim,
                        priority=floor.priority, solref=np.asarray(floor.solref).copy(),
                        solimp=np.asarray(floor.solimp).copy())
        for row in range(self.kinds.shape[0]):
            for col in range(self.kinds.shape[1]):
                name = f"motion_train_terrain_{row}_{col}"
                geom = floor if row == col == 0 else spec.worldbody.add_geom(name=name, **material)
                x0, y0 = border + row * cells, border + col * cells
                patch = self.heights[y0:y0 + cells + 1, x0:x0 + cells + 1]
                xy = self.origins[row, col, :2]
                if self.kinds[row, col] == 0:
                    geom.type = mujoco.mjtGeom.mjGEOM_BOX
                    geom.size = [PATCH_SIZE / 2, PATCH_SIZE / 2, .05]
                    geom.pos = [*xy, -.05]
                else:
                    low, high = float(patch.min()), float(patch.max())
                    extent = max(high - low, VERTICAL_SCALE)
                    spec.add_hfield(name=name, nrow=patch.shape[0], ncol=patch.shape[1],
                                    size=[PATCH_SIZE / 2, PATCH_SIZE / 2, extent, .10],
                                    userdata=((patch - low) / extent).ravel())
                    geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
                    geom.hfieldname = name
                    geom.pos = [*xy, low]
        width, length = 2 * self.half_size
        borders = ((0., ATLAS_BORDER, 0., length),
                   (width - ATLAS_BORDER, width, 0., length),
                   (ATLAS_BORDER, width - ATLAS_BORDER, 0., ATLAS_BORDER),
                   (ATLAS_BORDER, width - ATLAS_BORDER, length - ATLAS_BORDER, length))
        for i, (x0, x1, y0, y1) in enumerate(borders):
            spec.worldbody.add_geom(name=f"terrain_border_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                   size=[(x1 - x0) / 2, (y1 - y0) / 2, .05],
                                   pos=[(x0 + x1) / 2, (y0 + y1) / 2, -.05], **material)

    def sample_origins(self, key, count):
        """Startup origins: random row, shuffled evenly allocated columns."""
        rows, cols = self.kinds.shape
        row_key, col_key = jax.random.split(key)
        row = jax.random.randint(row_key, (count,), 0, rows)
        col = jnp.minimum(jnp.arange(count) * cols // count, cols - 1)
        col = jax.random.permutation(col_key, col)
        return jnp.asarray(self.origins)[row, col]

    def contains(self, xy, margin=0.):
        size = jnp.asarray(self.half_size)
        return jnp.all(jnp.abs(xy - size) <= size - margin, axis=-1)

    def height_at(self, xy):
        """Surface height for (..., 2) points using MuJoCo's triangle diagonal.

        The derivative within each triangle matches the physical terrain plane.
        Outside the finite atlas queries clamp to its flat border; callers can
        use ``contains`` to terminate worlds that leave the physical surface.
        """
        shape = np.array(self.heights.shape[::-1])
        grid = xy / HORIZONTAL_SCALE
        cell = jnp.clip(jnp.floor(grid).astype(jnp.int32), 0, jnp.asarray(shape - 2))
        fraction = jnp.clip(grid - cell, 0., 1.)
        x, y = cell[..., 0], cell[..., 1]
        fx, fy = fraction[..., 0], fraction[..., 1]
        h = jnp.asarray(self.heights)
        h00, h10, h01, h11 = h[y, x], h[y, x + 1], h[y + 1, x], h[y + 1, x + 1]
        lower = h00 + fx * (h10 - h00) + fy * (h11 - h10)
        upper = h00 + fx * (h11 - h01) + fy * (h01 - h00)
        return jnp.where(fx >= fy, lower, upper)
