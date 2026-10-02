import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1RunEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Official g1_with_hands.xml actuator order:
    #   0-14: legs + waist
    #   15-18: left shoulder + elbow
    #   19-28: left wrist + fingers
    #   29-32: right shoulder + elbow
    #   33-42: right wrist + fingers
    ACTIVE_ACTUATORS = np.array(
        list(range(0, 19)) + list(range(29, 33)), dtype=np.int32
    )
    LOCKED_ACTUATORS = np.array(
        list(range(19, 29)) + list(range(33, 43)), dtype=np.int32
    )

    # Indices within ACTIVE_ACTUATORS, not global actuator IDs.
    LEFT_SHOULDER_PITCH_ACTIVE_IDX = 15
    LEFT_ELBOW_ACTIVE_IDX = 18
    RIGHT_SHOULDER_PITCH_ACTIVE_IDX = 19
    RIGHT_ELBOW_ACTIVE_IDX = 22

    # Compiled geom/body IDs for the official scene_with_hands.xml.
    FLOOR_GEOM = 0
    LEFT_FOOT_GEOMS = {15, 16, 17, 18}
    RIGHT_FOOT_GEOMS = {30, 31, 32, 33}
    LEFT_ANKLE_BODY = 7
    RIGHT_ANKLE_BODY = 13
    STANDING_FOOT_Z = 0.033

    def __init__(self, render_mode=None, target_speed=2.0):
        super().__init__()
        self.render_mode = render_mode
        self.target_speed = float(target_speed)

        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_g1/scene_with_hands.xml")
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 43:
            raise ValueError(
                f"Expected 43 actuators from the official G1 hands model, "
                f"but found {self.model.nu}."
            )

        self.n_active = len(self.ACTIVE_ACTUATORS)
        ctrl_range = self.model.actuator_ctrlrange
        self.joint_low = ctrl_range[self.ACTIVE_ACTUATORS, 0].copy()
        self.joint_high = ctrl_range[self.ACTIVE_ACTUATORS, 1].copy()

        self.active_qpos_indices = []
        self.active_dof_indices = []
        for actuator_id in self.ACTIVE_ACTUATORS:
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            self.active_qpos_indices.append(self.model.jnt_qposadr[joint_id])
            self.active_dof_indices.append(self.model.jnt_dofadr[joint_id])
        self.active_qpos_indices = np.asarray(self.active_qpos_indices)
        self.active_dof_indices = np.asarray(self.active_dof_indices)

        # The action is a position offset around the official standing pose.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.active_ctrl_defaults = self.data.ctrl[
            self.ACTIVE_ACTUATORS
        ].copy()
        self.locked_ctrl_defaults = self.data.ctrl[
            self.LOCKED_ACTUATORS
        ].copy()
        self.default_active_qpos = self.data.qpos[self.active_qpos_indices].copy()
        self.standing_height = float(self.data.qpos[2])

        # Smaller ranges for balance joints, larger ranges for running swing.
        self.action_scale = np.full(self.n_active, 0.45, dtype=np.float64)
        self.action_scale[12:15] = 0.25  # waist
        self.action_scale[15:18] = (0.65, 0.35, 0.35)  # left shoulder
        self.action_scale[18] = 0.45  # left elbow
        self.action_scale[19:22] = (0.65, 0.35, 0.35)  # right shoulder
        self.action_scale[22] = 0.45  # right elbow

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_active,),
            dtype=np.float32,
        )

        # Observation follows the walking environment exactly, expanded from
        # 15 active joints to 23:
        # joints(23) + vel(23) + height(1) + roll/pitch(2)
        # + ang_vel(3) + lin_vel(3) + gait_clock(2) + fatigue(23) = 80.
        obs_dim = 3 * self.n_active + 11
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.frame_skip = 4
        self.gait_period = 40
        self.max_steps = 1000
        self.min_height = 0.52

        self.phase = 0.0
        self.step_count = 0
        self.previous_action = np.zeros(self.n_active, dtype=np.float64)

        # Gait clock state. The observation exposes sin/cos of this phase.
        self.gait_clock = np.zeros(2, dtype=np.float64)

        # Per-active-joint fatigue state, as in G1WalkEnv.
        self.fatigue = np.zeros(self.n_active, dtype=np.float64)
        self.fatigue_decay = 0.95
        self.fatigue_scale = 0.05

    def _get_active_joint_angles(self):
        return self.data.qpos[self.active_qpos_indices].copy()

    def _get_active_joint_velocities(self):
        return self.data.qvel[self.active_dof_indices].copy()

    def _get_root_roll_pitch(self):
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        return np.array([roll, pitch], dtype=np.float64)

    def _get_foot_contacts(self):
        left_contact = False
        right_contact = False

        for contact_id in range(self.data.ncon):
            geom1 = self.data.contact[contact_id].geom1
            geom2 = self.data.contact[contact_id].geom2

            if geom1 != self.FLOOR_GEOM and geom2 != self.FLOOR_GEOM:
                continue

            other_geom = geom2 if geom1 == self.FLOOR_GEOM else geom1
            if other_geom in self.LEFT_FOOT_GEOMS:
                left_contact = True
            if other_geom in self.RIGHT_FOOT_GEOMS:
                right_contact = True

        return left_contact, right_contact

    def _update_fatigue(self):
        # Measure sustained displacement from the standing control pose. This
        # avoids treating the non-zero elbow keyframe as permanent effort.
        effort = np.abs(
            self.data.ctrl[self.ACTIVE_ACTUATORS]
            - self.active_ctrl_defaults
        )
        self.fatigue = (
            self.fatigue_decay * self.fatigue
            + self.fatigue_scale * effort
        )

    def _update_gait_clock(self):
        self.gait_clock[0] = np.sin(self.phase)
        self.gait_clock[1] = np.cos(self.phase)

    def _get_obs(self):
        joint_offset = self._get_active_joint_angles() - self.default_active_qpos

        return np.concatenate(
            [
                joint_offset,
                self._get_active_joint_velocities(),
                [self.data.qpos[2]],
                self._get_root_roll_pitch(),
                self.data.qvel[3:6],
                self.data.qvel[0:3],
                self.gait_clock,
                self.fatigue,
            ]
        ).astype(np.float32)

    def _get_gait_rewards(self, left_contact, right_contact):
        """Reward alternating running contacts and short flight phases."""
        sin_phase = np.sin(self.phase)
        flight_band = 0.28

        if abs(sin_phase) <= flight_band:
            # Running contains a short period where neither foot is grounded.
            contact_reward = float(not left_contact and not right_contact)
            flight_reward = contact_reward
            swing_foot_z = min(
                self.data.xpos[self.LEFT_ANKLE_BODY, 2],
                self.data.xpos[self.RIGHT_ANKLE_BODY, 2],
            )
        elif sin_phase > 0.0:
            # Left stance, right swing.
            contact_reward = 0.5 * float(left_contact) + 0.5 * float(not right_contact)
            flight_reward = 0.0
            swing_foot_z = self.data.xpos[self.RIGHT_ANKLE_BODY, 2]
        else:
            # Right stance, left swing.
            contact_reward = 0.5 * float(right_contact) + 0.5 * float(not left_contact)
            flight_reward = 0.0
            swing_foot_z = self.data.xpos[self.LEFT_ANKLE_BODY, 2]

        target_clearance = 0.12
        clearance = np.clip(
            (swing_foot_z - self.STANDING_FOOT_Z) / target_clearance,
            0.0,
            1.0,
        )
        return float(contact_reward), float(flight_reward), float(clearance)

    def _get_arm_swing_reward(self):
        joint_offset = self._get_active_joint_angles() - self.default_active_qpos
        sin_phase = np.sin(self.phase)

        # Opposite arms move in opposite directions. If the viewer shows the
        # anatomical direction reversed, swap the signs of both targets.
        arm_amplitude = 0.45
        left_target = arm_amplitude * sin_phase
        right_target = -arm_amplitude * sin_phase

        left_shoulder = joint_offset[self.LEFT_SHOULDER_PITCH_ACTIVE_IDX]
        right_shoulder = joint_offset[self.RIGHT_SHOULDER_PITCH_ACTIVE_IDX]
        shoulder_error = (
            (left_shoulder - left_target) ** 2
            + (right_shoulder - right_target) ** 2
        )

        # A moderately flexed elbow is more natural and avoids straight-arm
        # swinging. Offsets are relative to the standing keyframe.
        left_elbow = joint_offset[self.LEFT_ELBOW_ACTIVE_IDX]
        right_elbow = joint_offset[self.RIGHT_ELBOW_ACTIVE_IDX]
        elbow_error = left_elbow**2 + right_elbow**2

        return float(np.exp(-4.0 * shoulder_error - 1.0 * elbow_error))

    def _get_reward(self, action):
        forward_velocity = float(self.data.qvel[0])
        lateral_velocity = float(self.data.qvel[1])
        vertical_velocity = float(self.data.qvel[2])
        left_contact, right_contact = self._get_foot_contacts()

        # Track a running speed instead of rewarding unlimited speed.
        speed_error = (forward_velocity - self.target_speed) / 0.8
        speed_reward = float(np.exp(-(speed_error**2)))
        forward_progress = float(np.clip(forward_velocity, -1.0, self.target_speed))

        gait_reward, flight_reward, clearance_reward = self._get_gait_rewards(
            left_contact, right_contact
        )
        arm_swing_reward = self._get_arm_swing_reward()

        height_target = self.standing_height - 0.04
        height_reward = float(
            np.exp(-12.0 * (float(self.data.qpos[2]) - height_target) ** 2)
        )

        roll, pitch = self._get_root_roll_pitch()
        target_pitch = 0.08
        posture_reward = float(
            np.exp(-6.0 * roll**2 - 4.0 * (pitch - target_pitch) ** 2)
        )

        lateral_penalty = -0.35 * lateral_velocity**2
        vertical_penalty = -0.05 * vertical_velocity**2
        fatigue_penalty = -0.05 * float(np.mean(self.fatigue**2))
        action_penalty = -0.01 * float(np.mean(action**2))
        action_rate_penalty = -0.03 * float(
            np.mean((action - self.previous_action) ** 2)
        )

        reward_terms = {
            "speed": 3.0 * speed_reward,
            "progress": 0.5 * forward_progress,
            "gait": 1.5 * gait_reward,
            "flight": 0.5 * flight_reward,
            "clearance": 0.8 * clearance_reward,
            "arm_swing": 0.4 * arm_swing_reward,
            "height": 0.5 * height_reward,
            "posture": 0.8 * posture_reward,
            "lateral_penalty": lateral_penalty,
            "vertical_penalty": vertical_penalty,
            "fatigue_penalty": fatigue_penalty,
            "action_penalty": action_penalty,
            "action_rate_penalty": action_rate_penalty,
        }
        return float(sum(reward_terms.values())), reward_terms

    def _is_terminated(self):
        if self.data.qpos[2] < self.min_height:
            return True
        if not np.all(np.isfinite(self.data.qpos)):
            return True

        roll, pitch = self._get_root_roll_pitch()
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        joint_noise = self.np_random.uniform(-0.015, 0.015, self.n_active)
        self.data.qpos[self.active_qpos_indices] += joint_noise
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.ACTIVE_ACTUATORS] = self.active_ctrl_defaults
        self.data.ctrl[self.LOCKED_ACTUATORS] = self.locked_ctrl_defaults
        mujoco.mj_forward(self.model, self.data)

        self.phase = self.np_random.uniform(0.0, 2.0 * np.pi)
        self._update_gait_clock()
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.fatigue.fill(0.0)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        target = (
            self.active_ctrl_defaults
            + self.action_scale * action
        )
        target = np.clip(target, self.joint_low, self.joint_high)

        self.data.ctrl[self.ACTIVE_ACTUATORS] = target
        self.data.ctrl[self.LOCKED_ACTUATORS] = self.locked_ctrl_defaults

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._update_fatigue()
        self.phase = (
            self.phase + 2.0 * np.pi / self.gait_period
        ) % (2.0 * np.pi)
        self._update_gait_clock()
        self.step_count += 1

        reward, reward_terms = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 10.0

        self.previous_action = action.copy()
        observation = self._get_obs()
        left_contact, right_contact = self._get_foot_contacts()
        info = {
            "forward_velocity": float(self.data.qvel[0]),
            "x_position": float(self.data.qpos[0]),
            "height": float(self.data.qpos[2]),
            "left_contact": left_contact,
            "right_contact": right_contact,
            "reward_terms": reward_terms,
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
