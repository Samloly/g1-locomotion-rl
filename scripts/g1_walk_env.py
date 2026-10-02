import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class G1WalkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps":50}

    ACTIVE_ACTUATORS = list(range(15))
    LOCKED_ACTUATORS = list(range(15,43))

    LEFT_FOOT_GEOMS = {15, 16, 17, 18}
    RIGHT_FOOT_GEOMS = {30, 31, 32, 33}
    FLOOR_GEOM = 0
    LEFT_ANKLE_BODY = 7
    RIGHT_ANKLE_BODY = 13
    STANDING_FOOT_Z = 0.033

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_g1/scene_with_hands.xml")
        self.data = mujoco.MjData(self.model)

        self.n_qpos = self.model.nq
        self.n_qvel = self.model.nv
        self.n_active = len(self.ACTIVE_ACTUATORS)

        ctrl_range = self.model.actuator_ctrlrange
        self.joint_low = ctrl_range[self.ACTIVE_ACTUATORS,0].copy()
        self.joint_high = ctrl_range[self.ACTIVE_ACTUATORS,1].copy()

        self.active_qpos_indices = []
        self.active_dof_indicies = []

        for act_idx in self.ACTIVE_ACTUATORS:
            jnt_id = self.model.actuator_trnid[act_idx][0]
            self.active_qpos_indices.append(self.model.jnt_qposadr[jnt_id])
            self.active_dof_indicies.append(self.model.jnt_dofadr[jnt_id])
        self.active_qpos_indices = np.array(self.active_qpos_indices)
        self.active_dof_indicies = np.array(self.active_dof_indicies)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.standing_height = self.data.qpos[2]

        self.locked_ctrl_defaults = np.zeros(len(self.LOCKED_ACTUATORS))
        for i,act_idx in enumerate(self.LOCKED_ACTUATORS):
            jnt_id = self.model.actuator_trnid[act_idx][0]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.locked_ctrl_defaults[i] = self.data.qpos[qpos_adr]

        self.min_height = 0.5

        # Gait clock: ~1.2 Hz cycle
        self.gait_period = 60
        self.phase = 0.0

        # Fatigue tracking
        self.fatigue = np.zeros(self.n_active)
        self.fatigue_decay = 0.95
        self.fatigue_scale = 0.05

        print(f"G1 Walk Final: {self.n_active} active joints")
        print(f"Standing height: {self.standing_height:.4f}m")

        # action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_active,), dtype=np.float32
        )

        # Observation: joints(15) + vel(15) + height(1) + roll/pitch(2)
        #   + ang_vel(3) + lin_vel(3) + gait_clock(2) + fatigue(15) = 56
        obs_dim = self.n_active + self.n_active + 1 + 2 + 3 + 3 + 2 + self.n_active
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        self.max_steps = 1000
        self.step_count = 0

    def _get_active_joint_angles(self):
        return self.data.qpos[self.active_qpos_indices].copy()

    def _get_active_joint_vel(self):
        return self.data.qvel[self.active_dof_indicies].copy()

    def _get_root_roll_pitch(self):
        w, x,y,z = self.data.qpos[3:7]
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2))
        pitch = np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        return np.array([roll,pitch])

    def _get_foot_contacts(self):
        left = False
        right = False
        for i in range(self.data.ncon):
            g1 = self.data.contact[i].geom1
            g2 = self.data.contact[i].geom2
            if g1 ==self.FLOOR_GEOM or g2==self.FLOOR_GEOM:
                other = g2 if g1 ==self.FLOOR_GEOM else g1
                if other in self.LEFT_FOOT_GEOMS:
                    left=True
                if other in self.RIGHT_FOOT_GEOMS:
                    right= True
        return left, right
    
    def _update_fatigue(self):
        effort = np.abs(self.data.ctrl[self.ACTIVE_ACTUATORS])
        self.fatigue = self.fatigue_decay*self.fatigue+self.fatigue_scale*effort

    def _get_obs(self):
        return np.concatenate([
            self._get_active_joint_angles(),
            self._get_active_joint_vel(),
            [self.data.qpos[2]],
            self._get_root_roll_pitch(),
            self.data.qvel[3:6],
            self.data.qvel[0:3],
            [np.sin(self.phase), np.cos(self.phase)],
            self.fatigue,
        ]).astype(np.float32)

    def _get_reward(self):
        forward_vel = self.data.qvel[0]
        left_contact, right_contact = self._get_foot_contacts()
        sin_phase = np.sin(self.phase)

        left_foot_z = self.data.xpos[self.LEFT_ANKLE_BODY][2]
        right_foot_z = self.data.xpos[self.RIGHT_ANKLE_BODY][2]

        # Forward velocity
        velocity_reward = forward_vel

        # Gait contact pattern
        gait_reward =0.0
        clearance_reward = 0.0
        target_clearance = 0.10

        if sin_phase>0:
            #left stance,right swing
            if left_contact:
                gait_reward +=0.5
            if not right_contact:
                gait_reward +=0.5
            swing_height = right_foot_z-self.STANDING_FOOT_Z
            clearance_reward = min(max(swing_height/target_clearance,0.0),1.0)
        else:
            # right stance, left swing
            if right_contact:
                gait_reward += 0.5
            if not left_contact:
                gait_reward += 0.5
            swing_height = left_foot_z - self.STANDING_FOOT_Z
            clearance_reward = min(max(swing_height / target_clearance, 0.0), 1.0)

        # stay upright
        height = self.data.qpos[2]
        height_reward = np.exp(-10.0 * (height-self.standing_height)**2)

        roll_pitch = self._get_root_roll_pitch()
        orientation_reward = np.exp(-5.0*np.sum(roll_pitch**2))

        # 4. Lateral velocity penalty
        lateral_penalty = -0.3 * self.data.qvel[1] ** 2

        # 5. Fatigue penalty
        fatigue_penalty = -0.1 * np.sum(self.fatigue ** 2)

        # 6. Double-flight penalty
        double_flight_penalty = 0.0
        if not left_contact and not right_contact:
            double_flight_penalty = -3.0

        # 7. Action smoothness: penalize large ctrl changes
        ctrl_penalty = -0.005 * np.mean(self.data.ctrl[self.ACTIVE_ACTUATORS] ** 2)

        reward = (2.0 * velocity_reward
                  + 2.0 * gait_reward
                  + 1.5 * clearance_reward
                  + 0.5 * height_reward
                  + 0.5 * orientation_reward
                  + lateral_penalty
                  + fatigue_penalty
                  + double_flight_penalty
                  + ctrl_penalty)
        return float(reward)

    def _is_terminated(self):
        if self.data.qpos[2] < self.min_height:
            return True
        if np.any(np.isnan(self.data.qpos)):
            return True
        roll_pitch = self._get_root_roll_pitch()
        if np.any(np.abs(roll_pitch)>1.05):
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        noise = self.np_random.uniform(-0.01, 0.01, size=self.n_active)
        self.data.qpos[self.active_qpos_indices] += noise

        self.data.ctrl[self.ACTIVE_ACTUATORS] = 0.0
        self.data.ctrl[self.LOCKED_ACTUATORS] = self.locked_ctrl_defaults

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = self.np_random.uniform(0, 2 * np.pi)
        self.fatigue = np.zeros(self.n_active)
        return self._get_obs(), {}
    
    def step(self,action):
        ctrl_targets = action*0.4
        ctrl_targets = np.clip(ctrl_targets, self.joint_low, self.joint_high)

        self.data.ctrl[self.ACTIVE_ACTUATORS] = ctrl_targets
        self.data.ctrl[self.LOCKED_ACTUATORS] = self.locked_ctrl_defaults

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        self._update_fatigue()

        self.phase+=2*np.pi/self.gait_period
        if self.phase >= 2 * np.pi:
            self.phase -= 2 * np.pi

        self.step_count += 1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward = 0.0

        return obs, reward, terminated, truncated, {}
    
    def close(self):
        pass
                