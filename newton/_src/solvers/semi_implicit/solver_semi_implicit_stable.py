# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semi-implicit solver with unconditionally stable joint forces.

The standard ``SolverSemiImplicit`` evaluates all joint forces (attachment
springs, PD targets, joint limits) explicitly, which requires
``kd * dt / I_min < 2`` for angular stability.  Bodies with tiny inertia
(e.g. wrist/ankle links of a humanoid) violate this criterion and cause
simulation blow-up.

``SolverSemiImplicitStable`` replaces ALL explicit joint forces with
implicit velocity corrections applied *after* integration.  The implicit
formula

    delta_w = dt * tau / (I + dt^2 * ke + dt * kd)

has a denominator that is always positive, making joint forces
unconditionally stable regardless of body mass or inertia.  All operations
are simple arithmetic with well-defined Warp adjoints, so the solver is
fully compatible with ``wp.Tape()`` for BPTT gradient computation.
"""

from __future__ import annotations

import warp as wp

from ...core import quat_twist
from ...core.types import override
from ...sim import Contacts, Control, JointType, Model, State
from .kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from .kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from .solver_semi_implicit import SolverSemiImplicit


# ---------------------------------------------------------------------------
# Apply FREE/DISTANCE joint wrenches to body_f (before integration)
# ---------------------------------------------------------------------------

@wp.kernel
def apply_free_joint_wrench(
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_f: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply user-specified wrench for FREE and DISTANCE joints only."""
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return

    if type != int(JointType.FREE) and type != int(JointType.DISTANCE):
        return

    c_child = joint_child[tid]
    qd_start = joint_qd_start[tid]

    wrench = wp.spatial_vector(
        joint_f[qd_start + 0],
        joint_f[qd_start + 1],
        joint_f[qd_start + 2],
        joint_f[qd_start + 3],
        joint_f[qd_start + 4],
        joint_f[qd_start + 5],
    )
    wp.atomic_add(body_f, c_child, wrench)


# ---------------------------------------------------------------------------
# Implicit joint force correction kernel (PD + limits + attachment)
# ---------------------------------------------------------------------------

@wp.func
def implicit_joint_force(
    q: float,
    qd: float,
    target_pos: float,
    target_vel: float,
    target_ke: float,
    target_kd: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
):
    """Compute joint force and effective stiffness/damping for implicit integration.

    Returns (force, effective_ke, effective_kd) as a vec3.
    When at a limit, PD targets are disabled and limit forces take over.
    """
    if q < limit_lower:
        f = limit_ke * (limit_lower - q) - limit_kd * qd
        return wp.vec3(f, limit_ke, limit_kd)
    elif q > limit_upper:
        f = limit_ke * (limit_upper - q) - limit_kd * qd
        return wp.vec3(f, limit_ke, limit_kd)
    else:
        f = target_ke * (target_pos - q) + target_kd * (target_vel - qd)
        return wp.vec3(f, target_ke, target_kd)


@wp.kernel
def implicit_joint_forces(
    # Post-integration body state (velocities corrected in-place)
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    # Model data
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    # Joint structure
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_child: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    # PD target parameters
    joint_f: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    # Joint limit parameters
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    # Attachment spring parameters
    joint_attach_ke: float,
    joint_attach_kd: float,
    dt: float,
):
    """Apply implicit joint force corrections to post-integration velocities.

    Handles ALL joint forces implicitly: PD targets, joint limits, and
    attachment springs.  This makes the solver unconditionally stable for
    any body mass or inertia.

    Launched with ``dim = model.joint_count``.
    """
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return

    # FREE and DISTANCE joints have no spring forces
    if type == int(JointType.FREE) or type == int(JointType.DISTANCE):
        return

    c_child = joint_child[tid]
    c_parent = joint_parent[tid]
    qd_start = joint_qd_start[tid]

    # ---------------------------------------------------------------
    # Compute kinematic state (mirrors eval_body_joints logic)
    # ---------------------------------------------------------------
    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    v_p = wp.vec3()
    w_p = wp.vec3()

    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(
            body_q[c_parent], body_com[c_parent]
        )
        twist_p = body_qd[c_parent]
        w_p = wp.spatial_bottom(twist_p)
        v_p = wp.spatial_top(twist_p) + wp.cross(w_p, r_p)

    X_wc = body_q[c_child] * X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(
        body_q[c_child], body_com[c_child]
    )
    twist_c = body_qd[c_child]
    w_c = wp.spatial_bottom(twist_c)
    v_c = wp.spatial_top(twist_c) + wp.cross(w_c, r_c)

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p) * q_c
    v_err = v_c - v_p
    w_err = w_c - w_p

    # ---------------------------------------------------------------
    # Compute forces and apply implicit corrections per joint type
    # ---------------------------------------------------------------
    angular_damping_scale = 0.01  # matches eval_body_joints

    # Accumulated corrections for child body
    delta_v_c = wp.vec3()
    delta_w_c = wp.vec3()

    m_c = body_mass[c_child]
    I_c = body_inertia[c_child]
    I_eff_c = wp.max(wp.min(I_c[0, 0], wp.min(I_c[1, 1], I_c[2, 2])), 1.0e-12)

    if type == int(JointType.REVOLUTE):
        axis = joint_axis[qd_start]
        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # Joint angle via swing-twist decomposition
        twist = quat_twist(axis, r_err)
        q_val = wp.acos(wp.clamp(twist[3], -1.0, 1.0)) * 2.0 * wp.sign(
            wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2]))
        )
        qd_val = wp.dot(w_err, axis_p)

        # --- PD + limit force along axis (implicit) ---
        fkd = implicit_joint_force(
            q_val,
            qd_val,
            joint_target_pos[qd_start],
            joint_target_vel[qd_start],
            joint_target_ke[qd_start],
            joint_target_kd[qd_start],
            joint_limit_lower[qd_start],
            joint_limit_upper[qd_start],
            joint_limit_ke[qd_start],
            joint_limit_kd[qd_start],
        )
        tau_pd = fkd[0]  # force value
        eff_ke = fkd[1]  # effective stiffness for denominator
        eff_kd = fkd[2]  # effective damping for denominator

        # Include user wrench
        tau_axis = -joint_f[qd_start] - tau_pd

        # Implicit correction along joint axis
        axis_denom = I_eff_c + dt * dt * eff_ke + dt * eff_kd
        delta_w_axis = dt * tau_axis / axis_denom
        delta_w_c += axis_p * delta_w_axis

        # --- Attachment forces (implicit) ---
        # Linear attachment
        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        lin_denom = m_c + dt * dt * joint_attach_ke + dt * joint_attach_kd
        if m_c > 0.0:
            delta_v_c += f_attach * (dt / lin_denom)

        # Angular attachment (off-axis swing)
        swing_err = wp.cross(axis_p, axis_c)
        t_attach = (
            swing_err * joint_attach_ke
            + (w_err - qd_val * axis_p) * joint_attach_kd * angular_damping_scale
        )
        # Moment-arm coupling from linear force
        total_t_attach = t_attach + wp.cross(r_c, f_attach)
        ang_attach_denom = (
            I_eff_c
            + dt * dt * joint_attach_ke
            + dt * joint_attach_kd * angular_damping_scale
        )
        delta_w_c += total_t_attach * (dt / ang_attach_denom)

    elif type == int(JointType.FIXED):
        # FIXED joints: all DOFs constrained by attachment
        ang_err = (
            wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2]))
            * wp.acos(wp.clamp(r_err[3], -1.0, 1.0))
            * 2.0
        )

        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        t_attach = (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke
            + w_err * joint_attach_kd * angular_damping_scale
        )

        lin_denom = m_c + dt * dt * joint_attach_ke + dt * joint_attach_kd
        if m_c > 0.0:
            delta_v_c += f_attach * (dt / lin_denom)

        total_t = t_attach + wp.cross(r_c, f_attach)
        ang_denom = (
            I_eff_c
            + dt * dt * joint_attach_ke
            + dt * joint_attach_kd * angular_damping_scale
        )
        delta_w_c += total_t * (dt / ang_denom)

    elif type == int(JointType.BALL):
        # BALL joints: linear attachment only, angular DOFs are free
        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        lin_denom = m_c + dt * dt * joint_attach_ke + dt * joint_attach_kd
        if m_c > 0.0:
            delta_v_c += f_attach * (dt / lin_denom)
            # Moment-arm coupling
            total_t = wp.cross(r_c, f_attach)
            ang_denom = (
                I_eff_c
                + dt * dt * joint_attach_ke
                + dt * joint_attach_kd
            )
            delta_w_c += total_t * (dt / ang_denom)

    elif type == int(JointType.PRISMATIC):
        axis = joint_axis[qd_start]
        axis_p = wp.transform_vector(X_wp, axis)

        # Joint displacement and velocity along axis
        q_val = wp.dot(x_err, axis_p)
        qd_val = wp.dot(v_err, axis_p)

        # --- PD + limit along prismatic axis (implicit) ---
        fkd = implicit_joint_force(
            q_val,
            qd_val,
            joint_target_pos[qd_start],
            joint_target_vel[qd_start],
            joint_target_ke[qd_start],
            joint_target_kd[qd_start],
            joint_limit_lower[qd_start],
            joint_limit_upper[qd_start],
            joint_limit_ke[qd_start],
            joint_limit_kd[qd_start],
        )
        f_pd = fkd[0]
        eff_ke = fkd[1]
        eff_kd = fkd[2]

        f_axis = -joint_f[qd_start] - f_pd
        lin_pd_denom = m_c + dt * dt * eff_ke + dt * eff_kd
        if m_c > 0.0:
            delta_v_c += axis_p * (dt * f_axis / lin_pd_denom)

        # --- Attachment (off-axis linear + full angular) ---
        ang_err = (
            wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2]))
            * wp.acos(wp.clamp(r_err[3], -1.0, 1.0))
            * 2.0
        )
        f_attach = (
            (x_err - q_val * axis_p) * joint_attach_ke
            + (v_err - qd_val * axis_p) * joint_attach_kd
        )
        t_attach = (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke
            + w_err * joint_attach_kd * angular_damping_scale
        )

        lin_attach_denom = m_c + dt * dt * joint_attach_ke + dt * joint_attach_kd
        if m_c > 0.0:
            delta_v_c += f_attach * (dt / lin_attach_denom)

        total_t = t_attach + wp.cross(r_c, f_attach)
        ang_denom = (
            I_eff_c
            + dt * dt * joint_attach_ke
            + dt * joint_attach_kd * angular_damping_scale
        )
        delta_w_c += total_t * (dt / ang_denom)

    # ---------------------------------------------------------------
    # Apply corrections to child body only
    # ---------------------------------------------------------------
    # Parent corrections are omitted: when a heavy child (e.g. torso,
    # I=0.034) is connected to a light parent (e.g. waist_roll, I=4e-6),
    # any inertia-ratio scaling amplifies corrections by 1000x+, causing
    # immediate blowup.  Correcting only the child body is standard in
    # position-based dynamics — the parent serves as the reference frame.
    if m_c > 0.0:
        wp.atomic_sub(body_qd, c_child, wp.spatial_vector(delta_v_c, delta_w_c))


# ---------------------------------------------------------------------------
# Re-integrate body positions from corrected velocities
# ---------------------------------------------------------------------------

@wp.kernel
def reintegrate_body_positions(
    body_q_old: wp.array(dtype=wp.transform),       # pre-integration positions
    body_qd_corrected: wp.array(dtype=wp.spatial_vector),  # post-correction velocities
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    dt: float,
    body_q_out: wp.array(dtype=wp.transform),       # output: corrected positions
):
    """Re-integrate body positions using corrected velocities.

    After ``implicit_joint_forces`` corrects ``body_qd``, the positions in
    ``body_q`` are stale (computed from pre-correction velocities).  This
    kernel recomputes ``x_new = x_com_old + v_corrected * dt`` and
    ``r_new = normalize(r_old + quat(w_corrected) * r_old * 0.5 * dt)``
    to restore position-velocity consistency.

    Launched with ``dim = model.body_count``.
    """
    tid = wp.tid()

    inv_mass = body_inv_mass[tid]

    # Skip fixed bodies (inv_mass == 0)
    if inv_mass == 0.0:
        return

    q_old = body_q_old[tid]
    x0 = wp.transform_get_translation(q_old)
    r0 = wp.transform_get_rotation(q_old)
    com = body_com[tid]

    # COM position before integration
    x_com = x0 + wp.quat_rotate(r0, com)

    # Corrected velocities (what will be state_in.body_qd next substep)
    qd = body_qd_corrected[tid]
    v1 = wp.spatial_top(qd)
    w1 = wp.spatial_bottom(qd)

    # Re-integrate position using corrected velocity (semi-implicit Euler)
    x1 = x_com + v1 * dt
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # Store as transform (body origin = COM position - rotated COM offset)
    body_q_out[tid] = wp.transform(x1 - wp.quat_rotate(r1, com), r1)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverSemiImplicitStable(SolverSemiImplicit):
    """Semi-implicit solver with unconditionally stable joint forces.

    Identical to :class:`SolverSemiImplicit` except that ALL joint forces
    (attachment springs, PD targets, joint limits) are applied via implicit
    velocity corrections *after* integration, rather than as explicit forces
    *before* integration.  This makes the solver unconditionally stable for
    any body mass or inertia.

    Use this solver when the model contains bodies with very small moments of
    inertia that cause the standard ``SolverSemiImplicit`` to diverge.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverSemiImplicitStable(model)

        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    def __init__(self, model: Model, **kwargs):
        super().__init__(model, **kwargs)
        self._debug = False
        self._debug_qd_buf = None

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            model = self.model

            if control is None:
                control = model.control(clone_variables=False)

            # --- Force accumulation (NO joint forces — all handled implicitly) ---

            # Damped springs
            eval_spring_forces(model, state_in, particle_f)

            # Triangle elastic and lift/drag
            eval_triangle_forces(model, state_in, control, particle_f)

            # Triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # Tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # Apply FREE/DISTANCE joint wrenches to body_f (constant forces, not springs)
            if model.joint_count and body_f is not None:
                wp.launch(
                    kernel=apply_free_joint_wrench,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_child,
                        model.joint_qd_start,
                        control.joint_f,
                        body_f,
                    ],
                    device=model.device,
                )

            # Particle-particle interactions
            eval_particle_contact_forces(model, state_in, particle_f)

            # Triangle/triangle contacts
            if self.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            # Body contacts
            eval_body_contact_forces(
                model, state_in, contacts, friction_smoothing=self.friction_smoothing
            )

            # Particle-body contacts
            eval_particle_body_contact_forces(
                model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=False
            )

            # --- Integration (same as parent) ---
            self.integrate_particles(model, state_in, state_out, dt)
            self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            # --- Save pre-correction state for debugging ---
            if self._debug:
                if self._debug_qd_buf is None:
                    self._debug_qd_buf = wp.zeros_like(state_out.body_qd)
                wp.copy(self._debug_qd_buf, state_out.body_qd)

            # --- Implicit joint force correction (ALL joint forces) ---
            if model.joint_count:
                wp.launch(
                    kernel=implicit_joint_forces,
                    dim=model.joint_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_child,
                        model.joint_parent,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_qd_start,
                        control.joint_f,
                        control.joint_target_pos,
                        control.joint_target_vel,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        model.joint_limit_kd,
                        self.joint_attach_ke,
                        self.joint_attach_kd,
                        dt,
                    ],
                    device=model.device,
                )

            # --- Re-integrate positions from corrected velocities ---
            if model.body_count:
                wp.launch(
                    kernel=reintegrate_body_positions,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_out.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        dt,
                    ],
                    outputs=[state_out.body_q],
                    device=model.device,
                )
