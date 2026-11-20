"""Training-only, paired adversarial recovery windows for the original G1 actor."""
import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env, nets
from .getup_amp import getup_frame
from .rot import mat_to_6d, mv3, tr3, yaw_mat


class G1PhaseDemos:
    """Condition an expert and a policy window on the same native clip, reset frame and end time."""

    def __init__(self, sim, demos, heading_features=False):
        ref = sim.reference
        if not sim.g1_reference or demos.clip_offsets is None or list(demos.files) != list(ref.files):
            raise ValueError("G1 phase AMP needs the same segmented bank for expert windows and floor reset references")
        self.sim = sim
        self.window = demos.window
        self.base_dim = sim.amp_dim + 2
        self.heading_features = heading_features
        self.xy = slice(self.base_dim + 2, self.base_dim + 4)
        self.raw_dim = self.xy.stop + (6 if heading_features else 0)
        self.condition_dim = 11
        self.frame_dim = self.raw_dim + self.condition_dim
        self.offsets = ref.offsets
        self.lengths = ref.lengths
        if not (np.array_equal(demos.clip_offsets, self.offsets)
                and np.array_equal(demos.clip_counts + self.window - 1, self.lengths)):
            raise ValueError("Reference and get-up discriminator clip frame indices differ")
        velocities = []
        for path in ref.files:
            with np.load(path) as d:
                velocities.append(d["root_lin_vel"])
        heading = yaw_mat(ref.rotation)
        v = mv3(tr3(heading), jnp.asarray(np.concatenate(velocities), jnp.float32))
        frames = [demos.frames, v[..., :2], ref.position[..., :2]]
        if heading_features:
            frames.append(mat_to_6d(ref.rotation))
        self.frames = jnp.concatenate(frames, axis=-1)
        # Inverse starting heading maps source-world motion into the horizontal
        # axes of the reference at reset. Use this identical transform for both labels.
        self.source_to_heading = tr3(heading)[..., :2, :2]
        if heading_features:
            # Use the more horizontal of the native pelvis's forward/left axes
            # at reset. Even with a vertical forward axis the 2-D basis is unit
            # length. It stays fixed for the episode and carries no policy gradients.
            forward = ref.rotation[:, :2, 0]
            left = ref.rotation[:, :2, 1]
            use_forward = (forward**2).sum(-1) >= (left**2).sum(-1)
            direction = jnp.where(use_forward[:, None], forward, jnp.stack((left[:, 1], -left[:, 0]), axis=-1))
            c, s = (direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)).T
            self.source_to_heading = jnp.stack((c, s, -s, c), axis=-1).reshape(-1, 2, 2)
        key_heights = demos.frames[:, sim.amp_dim - 12:sim.amp_dim].reshape(-1, 4, 3)[:, :, 2]
        self.start_pose = jnp.concatenate((ref.position[:, 2:3], -ref.rotation[:, 2, :], key_heights), axis=-1)

    def frame(self, state):
        """Native get-up AMP, horizontal velocity, world XY and optional world-frame body axes."""
        p, r, velocity, *_ = state
        heading_velocity = mv3(tr3(yaw_mat(r)), velocity)
        frames = [getup_frame(self.sim, state), heading_velocity[..., :2], p[..., :2]]
        if self.heading_features:
            frames.append(mat_to_6d(r))
        return jnp.concatenate(frames, axis=-1)

    def _sampling(self, clip, start, elapsed):
        clip, start, elapsed = jnp.broadcast_arrays(jnp.asarray(clip), jnp.asarray(start), jnp.asarray(elapsed))
        idx = jnp.clip(clip, 0, len(self.lengths)-1)
        last = self.lengths[idx] - 1
        end = start + elapsed
        valid = (idx == clip) & (start >= 0) & (elapsed >= self.window - 1) & (end <= last)
        # Invalid indices are clipped only to make gathers safe. The mask excludes
        # these windows from rewards, discriminator updates and running statistics.
        first = jnp.clip(start, 0, last)
        end = jnp.clip(end, 0, last)
        initial = self.offsets[idx] + first
        phase = jnp.stack((end / jnp.maximum(last, 1), first / jnp.maximum(last, 1), last * env.DT), axis=-1)
        condition = jnp.concatenate((phase, self.start_pose[initial]), axis=-1)
        return self.offsets[idx], initial, end, condition, valid

    def _expert(self, offset, end):
        frames = jnp.maximum(0, end[..., None] - (self.window - 1) + jnp.arange(self.window))
        return self.frames[offset[..., None] + frames]

    def _pack(self, raw, source_xy, initial, condition, yaw=None):
        if raw.shape[-2:] != (self.window, self.raw_dim):
            raise ValueError(f"Expected (..., {self.window}, {self.raw_dim}) G1 recovery windows")
        transform = self.source_to_heading[initial]
        if yaw is not None:
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            undo_yaw = jnp.stack((c, s, -s, c), axis=-1).reshape(c.shape + (2, 2))
            transform = transform @ undo_yaw
        transform = transform[..., None, :, :]
        xy = (transform @ source_xy[..., None])[..., 0]
        fields = [raw[..., :self.xy.start], xy]
        if self.heading_features:
            axes = raw[..., self.xy.stop:].reshape(*raw.shape[:-1], 3, 2)
            aligned = jnp.concatenate((transform @ axes[..., :2, :], axes[..., 2:, :]), axis=-2)
            fields.append(aligned.reshape(*raw.shape[:-1], 6))
        labels = jnp.broadcast_to(condition[..., None, :], raw.shape[:-1] + (self.condition_dim,))
        return jnp.concatenate((*fields, labels), axis=-1)

    def _robot(self, raw, yaw, initial, condition):
        # Floor reset puts the pelvis at world XY=(0,0) and rotates every world
        # vector by yaw. Undo that augmentation before comparison to native motion.
        return self._pack(raw, raw[..., self.xy], initial, condition, yaw=yaw)

    def condition_robot(self, windows, clip, start, yaw, elapsed):
        """Input for style scoring and its valid mask; elapsed is the clock of the window's final state."""
        _, initial, _, condition, valid = self._sampling(clip, start, elapsed)
        return self._robot(windows, yaw, initial, condition), valid

    def paired_windows(self, windows, clip, start, yaw, elapsed):
        """Positive and negative inputs with equal conditions; both exclude reset-crossing and out-of-clip windows."""
        offset, initial, end, condition, valid = self._sampling(clip, start, elapsed)
        expert = self._expert(offset, end)
        exy = expert[..., self.xy] - self.frames[initial, self.xy][..., None, :]
        return self._pack(expert, exy, initial, condition), self._robot(windows, yaw, initial, condition), valid

    def matched_feature_reward(self, disc, windows, clip, start, yaw, elapsed, scale=1.):
        """Bounded [0, scale] auxiliary similarity of learned first-layer LSGAN features.

        Distances are dimensionless relative to the reference window's RMS activation.
        The reward is exactly scale for an identical input and zero for invalid windows.
        The discriminator, normalization and expert branch are frozen. Derivatives
        through the policy window do not depend on the scalar LSGAN output's clamp.
        """
        expert, robot, valid = self.paired_windows(windows, clip, start, yaw, elapsed)
        norm = jax.tree.map(jax.lax.stop_gradient, disc["norm"])
        layer = jax.tree.map(jax.lax.stop_gradient, disc["net"][0])
        def features(w):
            x = amp.normalize(norm, w).reshape(*w.shape[:-2], -1)
            return jax.nn.leaky_relu(nets.mlp_apply([layer], x, jax.nn.leaky_relu))
        target = jax.lax.stop_gradient(features(expert))
        reference_rms = jnp.sqrt(jnp.mean(target**2, axis=-1, keepdims=True) + .1**2)
        difference = (features(robot) - target) / reference_rms
        # Pseudo-Huber distance has bounded slope for a single outlying hidden unit.
        distance = jnp.mean(jnp.sqrt(1. + difference**2) - 1., axis=-1)
        return jnp.where(valid, scale / (1. + distance / .2), 0.), valid

    def time_shifted_expert(self, clip, start, elapsed, shift_frames):
        """Wrong-time native negative with the positive's exact condition and reset origin.

        A candidate stays within this clip after its reset. It must move at least
        5 cm in height, 0.15 rad RMS over joints, or 8 cm RMS over the four relative
        landmarks versus the positive window. With orientation enabled, an RMS
        chord difference of 0.3 between corresponding body axes also qualifies.
        This excludes indistinguishable holds.
        The caller must apply this mask to the positive and negative as a pair.
        """
        offset, initial, end, condition, valid = self._sampling(clip, start, elapsed)
        _, _, wrong_end, _, shifted_valid = self._sampling(clip, start, elapsed + shift_frames)
        expert, shifted = self._expert(offset, end), self._expert(offset, wrong_end)
        delta = shifted - expert
        joints = jnp.mean(delta[..., :len(self.sim.amp_joints)]**2, axis=(-2, -1))
        points = delta[..., self.sim.amp_dim-12:self.sim.amp_dim].reshape(*delta.shape[:-1], 4, 3)
        landmarks = jnp.mean(jnp.sum(points**2, axis=-1), axis=(-2, -1))
        height = jnp.mean(delta[..., self.sim.amp_dim]**2, axis=-1)
        different = (height >= .05**2) | (joints >= .15**2) | (landmarks >= .08**2)
        if self.heading_features:
            direction = delta[..., self.xy.stop:].reshape(*delta.shape[:-1], 3, 2)
            # RMS chord distance of the two unit axes over the motion window.
            different |= (direction**2).sum(-2).mean(axis=(-2, -1)) >= .3**2
        xy = shifted[..., self.xy] - self.frames[initial, self.xy][..., None, :]
        return self._pack(shifted, xy, initial, condition), valid & shifted_valid & different

    def init(self, key, hidden=(1024, 512)):
        """Initialize the LSGAN and its frame normalizer on valid native windows, without padding."""
        st = amp.init(key, self.window, hidden, self.frame_dim)
        lengths = np.asarray(self.lengths)
        # Use actual eligible resets from each clip so the normalizer sees the
        # range of starting phases, displacements and initial posture markers.
        floor_clip = np.asarray(self.sim.floor["clip"])
        floor_frame = np.asarray(self.sim.floor["frame"])
        clip, starts, ends = [], [], []
        for i, n in enumerate(lengths):
            possible = np.sort(floor_frame[floor_clip == i])
            for end in range(self.window - 1, int(n)):
                count = np.searchsorted(possible, end - self.window + 1, side="right")
                if count:
                    clip.append(i)
                    starts.append(possible[(end - self.window + 1) % count])
                    ends.append(end)
        if not starts:
            raise ValueError("No G1 floor resets leave enough native frames for a phase AMP window")
        starts, ends = np.array(starts), np.array(ends)
        offset, initial, end, condition, _ = self._sampling(np.asarray(clip), starts, ends - starts)
        expert = self._expert(offset, end)
        xy = expert[..., self.xy] - self.frames[initial, self.xy][..., None, :]
        demo = self._pack(expert, xy, initial, condition)
        st["norm"] = nets.rn_update(st["norm"], demo)
        return st


def update_paired(st, demo, robot, valid, key, lr):
    """Equal positive/negative window origins. Empty batches leave all LSGAN and optimizer state intact."""
    if demo.shape != robot.shape or demo.ndim != 3 or valid.shape != (len(demo),) or not len(demo):
        raise ValueError("Expected (batch, window, features) paired arrays and one validity flag per window")
    valid = jnp.asarray(valid, bool)
    count = valid.sum()
    batch = len(demo)
    sample_key, update_key = jax.random.split(key)

    def train(_):
        def sample(_):
            ranks = jax.random.randint(sample_key, (batch,), 0, count)
            return jnp.searchsorted(jnp.cumsum(valid), ranks, side="right")
        ids = jax.lax.cond(count == batch, lambda _: jnp.arange(batch), sample, operand=None)
        return amp.train_step(st, demo[ids], robot[ids], jnp.ones(batch), update_key, lr, recovery=True)

    def skip(_):
        zero = jnp.zeros((), jnp.float32)
        return st, {"d_demo": zero, "d_robot": zero, "gp": zero}

    new, metrics = jax.lax.cond(count > 0, train, skip, operand=None)
    return new, {**metrics, "valid_fraction": count / batch}
