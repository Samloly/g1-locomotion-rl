"""Gymnasium environment for training Unitree G1 forward locomotion.

This is a compact, MuJoCo-CPU/SB3 adaptation of the observation and reward
ideas used by MuJoCo Playground's official G1 joystick task.  It preserves
the 23-actuator layout used by Samloly/g1-locomotion-rl.
"""

from __future__ import annotations

import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1RunEnv_new(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Official g1_with_hands.xml actuator ordering:
    # 0-14 legs + waist, 15-18 left shoulder/elbow,
    # 19-28 left wrist/fingers, 29-32 right shoulder/elbow,
    # 33-42 right wrist/fingers.
    ACTIVE_ACTUATORS = np.asarray(
        list(range(0, 19)) + list(range(29, 33)), dtype=np.int32
    )
    LOCKED_ACTUATORS = np.asarray(
        list(range(19, 29)) + list(range(33, 43)), dtype=np.int32
    )

    FLOOR_GEOM = 0
    LEFT_FOOT_GEOMS = {15, 16, 17, 18}
    RIGHT_FOOT_GEOMS = {30, 31, 32, 33}
    LEFT_ANKLE_BODY = 7
    RIGHT_ANKLE_BODY = 13
    STANDING_FOOT_Z = 0.033

    def __init__(
        self,
        render_mode=None,
        target_speed=4.0,
        model_path=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.target_speed = float(target_speed)

        if model_path is None:
            model_path = os.path.join(
                "mujoco_menagerie",
                "unitree_g1",
                "scene_with_hands.xml",
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 43:
            raise ValueError(
                "Expected 43 actuators from scene_with_hands.xml, "
                f"but found {self.model.nu}."
            )

        self.n_active = len(self.ACTIVE_ACTUATORS)
        self.frame_skip = 10
        self.max_steps = 1000
        self.min_height = 0.52

        # Official task uses 1.25-1.5 Hz.  A fixed 1.35 Hz is easier to debug.
        self.gait_frequency = 1.35
        self.phase = 0.0
        self.gait_clock = np.zeros(4, dtype=np.float64)

        self.step_count = 0
        self.previous_action = np.zeros(self.n_active, dtype=np.float64)

        self.keyframe_id = self._find_initial_keyframe()
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.keyframe_id
        )

        self.active_qpos_indices = []
        self.active_dof_indices = []
        for actuator_id in self.ACTIVE_ACTUATORS:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            self.active_qpos_indices.append(
                int(self.model.jnt_qposadr[joint_id])
            )
            self.active_dof_indices.append(
                int(self.model.jnt_dofadr[joint_id])
            )

        self.active_qpos_indices = np.asarray(
            self.active_qpos_indices, dtype=np.int32
        )
        self.active_dof_indices = np.asarray(
            self.active_dof_indices, dtype=np.int32
        )

        ctrl_range = self.model.actuator_ctrlrange
        self.joint_low = ctrl_range[self.ACTIVE_ACTUATORS, 0].copy()
        self.joint_high = ctrl_range[self.ACTIVE_ACTUATORS, 1].copy()

        self.active_ctrl_defaults = self.data.ctrl[
            self.ACTIVE_ACTUATORS
        ].copy()
        self.locked_ctrl_defaults = self.data.ctrl[
            self.LOCKED_ACTUATORS
        ].copy()
        self.default_active_qpos = self.data.qpos[
            self.active_qpos_indices
        ].copy()
        self.standing_height = float(self.data.qpos[2])

        self.pelvis_body_id = self._find_body_id(
            ("pelvis", "torso_link", "torso")
        )

        # Position offsets around the keyframe pose.
        self.action_scale = np.full(
            self.n_active, 0.45, dtype=np.float64
        )
        self.action_scale[12:15] = 0.25  # waist
        self.action_scale[15:18] = (0.65, 0.35, 0.35)
        self.action_scale[18] = 0.45
        self.action_scale[19:22] = (0.65, 0.35, 0.35)
        self.action_scale[22] = 0.45

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_active,),
            dtype=np.float32,
        )

        # local linvel(3), local angvel(3), projected gravity(3),
        # command(3), joint offset(23), joint vel(23), last action(23),
        # left/right phase sin/cos(4) = 85.
        obs_dim = 16 + 3 * self.n_active
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    @property
    def control_dt(self):
        return float(self.model.opt.timestep * self.frame_skip)

    def _find_initial_keyframe(self):
        for name in ("knees_bent", "stand", "home"):
            key_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, name
            )
            if key_id >= 0:
                return int(key_id)

        if self.model.nkey < 1:
            raise ValueError("The MuJoCo model contains no keyframe.")
        return 0

    def _find_body_id(self, candidates):
        for name in candidates:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id >= 0:
                return int(body_id)
        raise ValueError(
            "Could not find a pelvis/torso body. Tried: "
            + ", ".join(candidates)
        )

    def _get_active_joint_angles(self):
        return self.data.qpos[self.active_qpos_indices].copy()

    def _get_active_joint_velocities(self):
        return self.data.qvel[self.active_dof_indices].copy()

    def _get_body_velocity(self, body_id, local=False):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            int(local),
        )
        # MuJoCo spatial velocity ordering: angular first, linear second.
        return velocity[3:6].copy(), velocity[0:3].copy()

    def _get_local_linear_velocity(self):
        linear, _ = self._get_body_velocity(
            self.pelvis_body_id, local=True
        )
        return linear

    def _get_local_angular_velocity(self):
        _, angular = self._get_body_velocity(
            self.pelvis_body_id, local=True
        )
        return angular

    def _get_projected_gravity(self):
        rotation = self.data.xmat[
            self.pelvis_body_id
        ].reshape(3, 3)
        return rotation.T @ np.array(
            [0.0, 0.0, -1.0], dtype=np.float64
        )

    def _get_foot_contacts(self):
        left_contact = False
        right_contact = False

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 != self.FLOOR_GEOM and geom2 != self.FLOOR_GEOM:
                continue

            other_geom = geom2 if geom1 == self.FLOOR_GEOM else geom1
            left_contact |= other_geom in self.LEFT_FOOT_GEOMS
            right_contact |= other_geom in self.RIGHT_FOOT_GEOMS

        return bool(left_contact), bool(right_contact)

    def _update_gait_clock(self):
        left_phase = self.phase
        right_phase = self.phase + np.pi
        self.gait_clock[:] = (
            np.cos(left_phase),
            np.sin(left_phase),
            np.cos(right_phase),
            np.sin(right_phase),
        )

    def _get_obs(self):
        joint_offset = (
            self._get_active_joint_angles()
            - self.default_active_qpos
        )
        command = np.array(
            [self.target_speed, 0.0, 0.0], dtype=np.float64
        )

        observation = np.concatenate(
            (
                self._get_local_linear_velocity(),
                self._get_local_angular_velocity(),
                self._get_projected_gravity(),
                command,
                joint_offset,
                self._get_active_joint_velocities(),
                self.previous_action,
                self.gait_clock,
            )
        )
        return observation.astype(np.float32)

    def _get_feet_phase_reward(self):
        left_z = float(
            self.data.xpos[self.LEFT_ANKLE_BODY, 2]
        )
        right_z = float(
            self.data.xpos[self.RIGHT_ANKLE_BODY, 2]
        )

        swing_height = 0.10
        left_target = (
            self.STANDING_FOOT_Z
            + swing_height * max(np.sin(self.phase), 0.0)
        )
        right_target = (
            self.STANDING_FOOT_Z
            + swing_height * max(-np.sin(self.phase), 0.0)
        )

        error = (
            (left_z - left_target) ** 2
            + (right_z - right_target) ** 2
        )
        return float(np.exp(-error / 0.01))

    def _get_feet_slip_cost(self, left_contact, right_contact):
        left_linear, _ = self._get_body_velocity(
            self.LEFT_ANKLE_BODY, local=False
        )
        right_linear, _ = self._get_body_velocity(
            self.RIGHT_ANKLE_BODY, local=False
        )

        cost = 0.0
        if left_contact:
            cost += float(np.sum(left_linear[:2] ** 2))
        if right_contact:
            cost += float(np.sum(right_linear[:2] ** 2))
        return cost

    def _get_reward(self, action):
        local_linear = self._get_local_linear_velocity()
        local_angular = self._get_local_angular_velocity()

        velocity_error = (
            (float(local_linear[0]) - self.target_speed) ** 2
            + float(local_linear[1]) ** 2
        )
        tracking_velocity = float(
            np.exp(-velocity_error / 0.25)
        )

        # The official task compares projected gravity rather than Euler angles.
        projected_gravity = self._get_projected_gravity()
        target_gravity = np.array(
            [0.05, 0.0, -1.0], dtype=np.float64
        )
        orientation_cost = float(
            np.sum((projected_gravity - target_gravity) ** 2)
        )

        angular_velocity_cost = float(
            np.sum(local_angular[:2] ** 2)
        )

        left_contact, right_contact = self._get_foot_contacts()
        feet_phase_reward = self._get_feet_phase_reward()
        feet_slip_cost = self._get_feet_slip_cost(
            left_contact, right_contact
        )

        joint_offset = (
            self._get_active_joint_angles()
            - self.default_active_qpos
        )
        waist_offset = joint_offset[12:15]
        arm_offset = joint_offset[15:23]
        pose_cost = float(
            0.5 * np.mean(waist_offset**2)
            + 0.1 * np.mean(arm_offset**2)
        )

        action_rate_cost = float(
            np.mean((action - self.previous_action) ** 2)
        )

        reward_terms = {
            "tracking_velocity": 2.0 * tracking_velocity,
            "feet_phase": 0.75 * feet_phase_reward,
            "orientation_penalty": -1.0 * orientation_cost,
            "angular_velocity_penalty": -0.10
            * angular_velocity_cost,
            "feet_slip_penalty": -0.10 * feet_slip_cost,
            "pose_penalty": -0.10 * pose_cost,
            "action_rate_penalty": -0.005 * action_rate_cost,
        }
        return float(sum(reward_terms.values())), reward_terms

    def _is_terminated(self):
        if self.data.qpos[2] < self.min_height:
            return True
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        if not np.all(np.isfinite(self.data.qvel)):
            return True

        # Terminate when the pelvis is more than 90 degrees from upright.
        projected_gravity = self._get_projected_gravity()
        if projected_gravity[2] > 0.0:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del options

        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.keyframe_id
        )

        joint_noise = self.np_random.uniform(
            -0.015, 0.015, self.n_active
        )
        self.data.qpos[self.active_qpos_indices] += joint_noise
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.ACTIVE_ACTUATORS] = (
            self.active_ctrl_defaults
        )
        self.data.ctrl[self.LOCKED_ACTUATORS] = (
            self.locked_ctrl_defaults
        )
        mujoco.mj_forward(self.model, self.data)

        # Random phase prevents the policy from memorizing one reset pose.
        self.phase = float(
            self.np_random.uniform(0.0, 2.0 * np.pi)
        )
        self._update_gait_clock()
        self.step_count = 0
        self.previous_action.fill(0.0)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n_active,):
            raise ValueError(
                f"Expected action shape {(self.n_active,)}, "
                f"got {action.shape}."
            )
        action = np.clip(action, -1.0, 1.0)

        target = (
            self.active_ctrl_defaults
            + self.action_scale * action
        )
        target = np.clip(target, self.joint_low, self.joint_high)

        self.data.ctrl[self.ACTIVE_ACTUATORS] = target
        self.data.ctrl[self.LOCKED_ACTUATORS] = (
            self.locked_ctrl_defaults
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        reward, reward_terms = self._get_reward(action)
        terminated = self._is_terminated()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 10.0

        # Observation must contain the action that produced the current state.
        self.previous_action = action.copy()

        self.phase = (
            self.phase
            + 2.0
            * np.pi
            * self.gait_frequency
            * self.control_dt
        ) % (2.0 * np.pi)
        self._update_gait_clock()

        observation = self._get_obs()
        local_velocity = self._get_local_linear_velocity()
        left_contact, right_contact = self._get_foot_contacts()

        info = {
            "forward_velocity": float(local_velocity[0]),
            "lateral_velocity": float(local_velocity[1]),
            "x_position": float(self.data.qpos[0]),
            "height": float(self.data.qpos[2]),
            "left_contact": left_contact,
            "right_contact": right_contact,
            "reward_terms": reward_terms,
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

