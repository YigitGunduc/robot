from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from gear_sonic_mjx.g1_parameters import G1_MUJOCO_JOINT_NAMES


@dataclass
class G1IndexMap:
    qpos: torch.Tensor
    qvel: torch.Tensor
    actuator: torch.Tensor
    root_qpos_adr: int
    root_dof_adr: int


class MjWarpBatchSim:
    """Direct MuJoCo-Warp batched simulator with zero-copy Warp<->Torch views.

    This deliberately uses `mujoco_warp` instead of putting JAX between the simulator and a PyTorch
    PPO policy. MuJoCo-Warp is still MuJoCo's GPU implementation; Warp exposes zero-copy PyTorch
    views for its device arrays.

    Assumptions for SONIC-style control:
      * a floating-base G1 with one free joint;
      * the 29 canonical G1 joints are present by name;
      * 29 actuators map one-to-one to those joints;
      * actuator `ctrl` is interpreted as torque/effort. If your MJCF uses position actuators,
        either switch them to motors or use your existing actuator path instead of `write_torque`.
    """

    def __init__(self, mjcf_path: str | Path, nworld: int = 4096, timestep: float = 0.005, nconmax: int | None = None, njmax: int | None = None):
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError(
                "MuJoCo-Warp not installed. Install `pip install mujoco-mjx[warp] warp-lang` "
                "or a matching current MuJoCo/mujoco-warp build."
            ) from exc
        self.mujoco, self.mjw, self.wp = mujoco, mjw, wp
        self.mjcf_path = str(mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.mj_model.opt.timestep = float(timestep)
        self.index = self._build_index_map()
        self.model = mjw.put_model(self.mj_model)
        kwargs = {"nworld": int(nworld)}
        if nconmax is not None:
            kwargs["nconmax"] = int(nconmax)
        if njmax is not None:
            kwargs["njmax"] = int(njmax)
        self.data = mjw.make_data(self.mj_model, **kwargs)
        self.nworld = int(nworld)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.xpos = wp.to_torch(self.data.xpos) if hasattr(self.data, "xpos") else None
        self.xquat = wp.to_torch(self.data.xquat) if hasattr(self.data, "xquat") else None
        self.cvel = wp.to_torch(self.data.cvel) if hasattr(self.data, "cvel") else None
        self._qpos_idx = self.index.qpos.to(self.qpos.device)
        self._qvel_idx = self.index.qvel.to(self.qvel.device)
        self._act_idx = self.index.actuator.to(self.ctrl.device)

    def _build_index_map(self) -> G1IndexMap:
        m, mujoco = self.mj_model, self.mujoco
        free_ids = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if len(free_ids) != 1:
            raise ValueError(f"Expected exactly one floating-base free joint, found {len(free_ids)}")
        root = free_ids[0]
        qpos, qvel = [], []
        for name in G1_MUJOCO_JOINT_NAMES:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"G1 MJCF missing joint {name!r}")
            qpos.append(int(m.jnt_qposadr[jid]))
            qvel.append(int(m.jnt_dofadr[jid]))

        # Map each named joint to its actuator through actuator_trnid[:,0].
        act_by_joint = {}
        for aid in range(m.nu):
            jid = int(m.actuator_trnid[aid, 0])
            if jid >= 0:
                act_by_joint[jid] = aid
        acts = []
        for name in G1_MUJOCO_JOINT_NAMES:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid not in act_by_joint:
                raise KeyError(f"No actuator directly attached to {name!r}")
            acts.append(act_by_joint[jid])
        return G1IndexMap(
            torch.tensor(qpos, dtype=torch.long), torch.tensor(qvel, dtype=torch.long), torch.tensor(acts, dtype=torch.long),
            int(m.jnt_qposadr[root]), int(m.jnt_dofadr[root]),
        )

    @property
    def device(self) -> torch.device:
        return self.qpos.device

    def root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.index.root_qpos_adr
        return self.qpos[:, a:a+3], self.qpos[:, a+3:a+7]

    def root_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.index.root_dof_adr
        return self.qvel[:, a:a+3], self.qvel[:, a+3:a+6]

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qpos.index_select(1, self._qpos_idx), self.qvel.index_select(1, self._qvel_idx)


    def body_ids(self, names: list[str]) -> torch.Tensor:
        ids = []
        for name in names:
            bid = self.mujoco.mj_name2id(self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"MJCF missing body {name!r}")
            ids.append(bid)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def body_state(self, body_ids: torch.Tensor) -> dict[str, torch.Tensor | None]:
        pos = None if self.xpos is None else self.xpos.index_select(1, body_ids.to(self.xpos.device))
        quat = None if self.xquat is None else self.xquat.index_select(1, body_ids.to(self.xquat.device))
        linvel = angvel = None
        if self.cvel is not None:
            cv = self.cvel.index_select(1, body_ids.to(self.cvel.device))
            # MuJoCo cvel convention stores angular then linear spatial velocity.
            angvel, linvel = cv[..., :3], cv[..., 3:6]
        return {"body_pos": pos, "body_quat_wxyz": quat, "body_linvel": linvel, "body_angvel": angvel}

    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        lows, highs = [], []
        for name in G1_MUJOCO_JOINT_NAMES:
            jid = self.mujoco.mj_name2id(self.mj_model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            lows.append(float(self.mj_model.jnt_range[jid, 0])); highs.append(float(self.mj_model.jnt_range[jid, 1]))
        return torch.tensor(lows, device=self.device), torch.tensor(highs, device=self.device)

    @torch.no_grad()
    def set_state(self, env_ids: torch.Tensor, root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor | None = None, root_velocity6: torch.Tensor | None = None) -> None:
        env_ids = env_ids.to(self.qpos.device, torch.long)
        a, d = self.index.root_qpos_adr, self.index.root_dof_adr
        self.qpos[env_ids, a:a+3] = root_pos.to(self.qpos)
        self.qpos[env_ids, a+3:a+7] = root_quat_wxyz.to(self.qpos)
        self.qpos[env_ids[:, None], self._qpos_idx[None]] = joint_pos.to(self.qpos)
        self.qvel[env_ids] = 0.0
        if root_velocity6 is not None:
            self.qvel[env_ids, d:d+6] = root_velocity6.to(self.qvel)
        if joint_vel is not None:
            self.qvel[env_ids[:, None], self._qvel_idx[None]] = joint_vel.to(self.qvel)
        # Recompute derived quantities/contact state after externally changing qpos/qvel.
        self.mjw.forward(self.model, self.data)


    @torch.no_grad()
    def add_root_velocity(self, env_ids: torch.Tensor, delta6: torch.Tensor) -> None:
        env_ids = env_ids.to(self.qvel.device, torch.long)
        a = self.index.root_dof_adr
        self.qvel[env_ids, a:a+6] += delta6.to(self.qvel)

    def configure_startup_domain_randomization(
        self,
        mass_body_names: list[str] | None = None,
        mass_scale: tuple[float, float] = (0.8, 2.5),
        static_friction: tuple[float, float] = (0.3, 1.6),
        dynamic_friction: tuple[float, float] = (0.3, 1.2),
        num_variants: int = 64,
        seed: int = 0,
    ) -> None:
        """Create safe compiled physics variants and assign them across worlds.

        NVIDIA randomizes physical properties per environment. For direct MJWarp, mass changes have
        dependent model constants, so this implementation does *not* mutate `body_mass` alone. It
        compiles a configurable number of CPU MuJoCo variants, then batches all dependent fields
        documented by MJWarp and assigns variants across worlds. This is a close GPU-safe analogue
        that avoids 4096 separate model compilations.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        names = mass_body_names or []
        body_ids = []
        for name in names:
            bid = self.mujoco.mj_name2id(self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0: raise KeyError(name)
            body_ids.append(bid)
        variants = []
        for _ in range(max(1, int(num_variants))):
            m = self.mujoco.MjModel.from_xml_path(self.mjcf_path)
            # Friction convention: sliding, torsional, rolling. Keep torsional/rolling from XML.
            sf = rng.uniform(*static_friction, size=m.ngeom)
            df = rng.uniform(*dynamic_friction, size=m.ngeom)
            # MuJoCo has one sliding coefficient rather than separate static/dynamic Coulomb terms;
            # use a value between sampled dynamic/static limits and record this approximation.
            m.geom_friction[:, 0] = np.minimum(sf, np.maximum(df, 1e-4))
            for bid in body_ids:
                m.body_mass[bid] *= rng.uniform(*mass_scale)
            # Recompute constants that depend on mass/inertia.
            d = self.mujoco.MjData(m)
            self.mujoco.mj_setConst(m, d)
            variants.append(m)
        assignment = rng.integers(0, len(variants), size=self.nworld)

        def stack(field):
            return np.stack([getattr(variants[v], field) for v in assignment], axis=0)
        wp = self.wp
        # Fields required by MJWarp's documented per-world rigid-body variant contract.
        self.model.body_mass = wp.array(stack("body_mass"), dtype=float)
        self.model.body_subtreemass = wp.array(stack("body_subtreemass"), dtype=float)
        self.model.body_inertia = wp.array(stack("body_inertia"), dtype=wp.vec3)
        self.model.body_invweight0 = wp.array(stack("body_invweight0"), dtype=wp.vec2)
        self.model.body_ipos = wp.array(stack("body_ipos"), dtype=wp.vec3)
        self.model.body_iquat = wp.array(stack("body_iquat"), dtype=wp.quat)
        # geom_friction is a vec3 model field and accepts the same leading world dimension in MJWarp.
        self.model.geom_friction = wp.array(stack("geom_friction"), dtype=wp.vec3)

    @torch.no_grad()
    def write_torque(self, torque_mj: torch.Tensor) -> None:
        if torque_mj.shape != (self.nworld, 29):
            raise ValueError(f"Expected torque [{self.nworld},29], got {tuple(torque_mj.shape)}")
        self.ctrl[:, self._act_idx] = torque_mj.to(self.ctrl)

    @torch.no_grad()
    def step(self, substeps: int = 1) -> None:
        for _ in range(int(substeps)):
            self.mjw.step(self.model, self.data)
