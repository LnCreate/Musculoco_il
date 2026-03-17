class TorqueFidelityReward(object):
    """Phase-aware torque fidelity prior without external torque labels.

    Encourages:
    1) lower high-frequency torque jitter (smoothness),
    2) low torque magnitudes during swing,
    3) sufficient sagittal support torques during stance,
    4) bilateral consistency in double support.
    """

    def __init__(self,
                 body_mass_kg=75.0,
                 reward_scale=0.1,
                 smooth_weight=0.02,
                 swing_weight=1.0,
                 stance_weight=0.6,
                 balance_weight=0.2,
                 stance_grf_ratio=0.05,
                 swing_cap_hip=0.80,
                 swing_cap_knee=0.70,
                 swing_cap_ankle=0.55,
                 stance_floor_hip=0.15,
                 stance_floor_knee=0.12,
                 stance_floor_ankle=0.16):
        self.body_mass_kg = float(body_mass_kg)
        self.reward_scale = float(reward_scale)

        self.smooth_weight = float(smooth_weight)
        self.swing_weight = float(swing_weight)
        self.stance_weight = float(stance_weight)
        self.balance_weight = float(balance_weight)

        self.stance_grf_ratio = float(stance_grf_ratio)

        self.swing_caps = np.array([swing_cap_hip, swing_cap_knee, swing_cap_ankle], dtype=np.float64)
        self.stance_floors = np.array([stance_floor_hip, stance_floor_knee, stance_floor_ankle], dtype=np.float64)

        self._model = None
        self._data = None
        self._torque_idx = None
        self._prev_tau = None

    def set_mdp(self, mdp):
        self._model = mdp._model
        self._data = mdp._data

        def _joint_dof_idx(joint_name):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"Joint not found for torque fidelity reward: {joint_name}")
            return int(self._model.jnt_dofadr[jid])

        idx = [
            _joint_dof_idx("hip_flexion_r"),
            _joint_dof_idx("knee_angle_r"),
            _joint_dof_idx("ankle_angle_r"),
            _joint_dof_idx("hip_flexion_l"),
            _joint_dof_idx("knee_angle_l"),
            _joint_dof_idx("ankle_angle_l"),
        ]
        self._torque_idx = np.asarray(idx, dtype=np.int32)

    def reset(self):
        self._prev_tau = None

    @staticmethod
    def _hinge_sq(x):
        z = np.maximum(0.0, x)
        return z * z

    def __call__(self, state, action, next_state):
        if self._model is None or self._data is None or self._torque_idx is None:
            return 0.0

        qfrc_act = np.asarray(self._data.qfrc_actuator, dtype=np.float64)
        tau = qfrc_act[self._torque_idx] / max(1e-6, self.body_mass_kg)

        tau_r = np.abs(tau[:3])
        tau_l = np.abs(tau[3:])

        grf_l = _get_grf(self._model, self._data, LEFT_FOOT_BODIES)
        grf_r = _get_grf(self._model, self._data, RIGHT_FOOT_BODIES)
        bw = self.body_mass_kg * 9.81
        fz_l = abs(float(grf_l[2]))
        fz_r = abs(float(grf_r[2]))

        thr = self.stance_grf_ratio * bw
        left_stance = fz_l >= thr
        right_stance = fz_r >= thr

        swing_pen = 0.0
        stance_pen = 0.0

        if right_stance:
            stance_pen += float(np.mean(self._hinge_sq(self.stance_floors - tau_r)))
        else:
            swing_pen += float(np.mean(self._hinge_sq(tau_r - self.swing_caps)))

        if left_stance:
            stance_pen += float(np.mean(self._hinge_sq(self.stance_floors - tau_l)))
        else:
            swing_pen += float(np.mean(self._hinge_sq(tau_l - self.swing_caps)))

        smooth_pen = 0.0
        if self._prev_tau is not None:
            smooth_pen = float(np.mean((tau - self._prev_tau) ** 2))

        balance_pen = 0.0
        if left_stance and right_stance:
            balance_pen = float(np.mean((tau_r - tau_l) ** 2))

        self._prev_tau = tau.copy()

        total_pen = (
            self.smooth_weight * smooth_pen
            + self.swing_weight * swing_pen
            + self.stance_weight * stance_pen
            + self.balance_weight * balance_pen
        )

        return -self.reward_scale * float(total_pen)
