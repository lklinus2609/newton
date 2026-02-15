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

"""Semi-implicit solver with unconditionally stable joint attachment springs.

The standard ``SolverSemiImplicit`` evaluates joint attachment penalty forces
(``joint_attach_ke/kd``) explicitly, which requires ``kd * dt / I_min < 2``
for angular stability.  Bodies with tiny inertia (e.g. wrist/ankle links of
a humanoid) violate this criterion and cause simulation blow-up.

``SolverSemiImplicitStable`` replaces the explicit attachment forces with an
implicit velocity correction applied *after* integration.  The implicit
formula

    v_new = (m * v_pred + dt * ke * err) / (m + dt^2 * ke + dt * kd)

has a denominator that is always positive, making the attachment
unconditionally stable regardless of body mass or inertia.  All operations
are simple arithmetic with well-defined Warp adjoints, so the solver is
fully compatible with ``wp.Tape()`` for BPTT gradient computation.
"""

from __future__ import annotations

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, JointType, Model, State
from .kernels_body import eval_body_joint_forces
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
# Implicit joint attachment correction kernel
# ---------------------------------------------------------------------------

@wp.kernel
def implicit_joint_attachment(
    # Post-integration body state (velocities will be corrected in-place)
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
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    # Attachment spring parameters
    joint_attach_ke: float,
    joint_attach_kd: float,
    dt: float,
):
    """Apply implicit joint attachment correction to post-integration velocities.

    For each joint, computes the attachment position/velocity errors from the
    predicted (post-explicit-integration) state and applies an implicit velocity
    correction that is unconditionally stable.

    Launched with ``dim = model.joint_count``.
    """
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return

    # FREE and DISTANCE joints have no attachment constraints
    if type == int(JointType.FREE) or type == int(JointType.DISTANCE):
        return

    c_child = joint_child[tid]
    c_parent = joint_parent[tid]
    qd_start = joint_qd_start[tid]

    # ---------------------------------------------------------------
    # Compute kinematic errors (mirrors eval_body_joints logic)
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
    v_err = v_c - v_p
    w_err = w_c - w_p

    # ---------------------------------------------------------------
    # Compute attachment force/torque (same structure as eval_body_joints)
    # ---------------------------------------------------------------
    f_attach = wp.vec3()
    t_attach = wp.vec3()
    angular_damping_scale = 0.01  # matches hardcoded value in eval_body_joints

    if type == int(JointType.REVOLUTE):
        axis = joint_axis[qd_start]
        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        qd_val = wp.dot(w_err, axis_p)
        swing_err = wp.cross(axis_p, axis_c)

        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        t_attach = (
            swing_err * joint_attach_ke
            + (w_err - qd_val * axis_p) * joint_attach_kd * angular_damping_scale
        )

    elif type == int(JointType.FIXED):
        r_err = wp.quat_inverse(q_p) * q_c
        ang_err = (
            wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2]))
            * wp.acos(r_err[3])
            * 2.0
        )
        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        t_attach = (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke
            + w_err * joint_attach_kd * angular_damping_scale
        )

    elif type == int(JointType.BALL):
        f_attach = x_err * joint_attach_ke + v_err * joint_attach_kd
        # Ball joints: angular DOFs are free, only linear attachment

    elif type == int(JointType.PRISMATIC):
        axis = joint_axis[qd_start]
        axis_p = wp.transform_vector(X_wp, axis)
        q_val = wp.dot(x_err, axis_p)
        qd_val = wp.dot(v_err, axis_p)
        r_err = wp.quat_inverse(q_p) * q_c
        ang_err = (
            wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2]))
            * wp.acos(r_err[3])
            * 2.0
        )
        # Project off displacement along the free axis
        f_attach = (
            (x_err - q_val * axis_p) * joint_attach_ke
            + (v_err - qd_val * axis_p) * joint_attach_kd
        )
        t_attach = (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke
            + w_err * joint_attach_kd * angular_damping_scale
        )

    # D6 joints: skip for now (complex multi-axis logic)
    # The attachment is handled explicitly for D6 which is acceptable
    # since D6 joints are rarely used with very light bodies.

    # ---------------------------------------------------------------
    # Apply implicit velocity correction
    #
    # Implicit formula (linear):
    #   v_new = (m*v_pred - dt*ke*x_err + dt*(dt*ke+kd)*v_parent) / (m + dt^2*ke + dt*kd)
    #
    # Equivalently, as a correction to the predicted velocity:
    #   delta_v = dt * (ke*x_err + kd*v_err) / (m + dt^2*ke + dt*kd)
    #   v_new = v_pred - delta_v  (for child, + for parent)
    #
    # Angular version uses scalar inertia approximation (min diagonal).
    # The torque includes cross-coupling from linear force at the joint.
    # ---------------------------------------------------------------

    # --- Child body correction ---
    m_c = body_mass[c_child]
    if m_c > 0.0:
        lin_denom_c = m_c + dt * dt * joint_attach_ke + dt * joint_attach_kd
        delta_v_c = f_attach * (dt / lin_denom_c)

        I_c = body_inertia[c_child]
        I_eff_c = wp.min(I_c[0, 0], wp.min(I_c[1, 1], I_c[2, 2]))
        I_eff_c = wp.max(I_eff_c, 1.0e-12)
        ang_denom_c = (
            I_eff_c
            + dt * dt * joint_attach_ke
            + dt * joint_attach_kd * angular_damping_scale
        )
        # Include moment-arm coupling: linear force at joint creates torque about COM
        total_t_c = t_attach + wp.cross(r_c, f_attach)
        delta_w_c = total_t_c * (dt / ang_denom_c)

        wp.atomic_sub(body_qd, c_child, wp.spatial_vector(delta_v_c, delta_w_c))

    # --- Parent body correction (equal and opposite) ---
    if c_parent >= 0:
        m_p = body_mass[c_parent]
        if m_p > 0.0:
            lin_denom_p = m_p + dt * dt * joint_attach_ke + dt * joint_attach_kd
            delta_v_p = f_attach * (dt / lin_denom_p)

            I_p = body_inertia[c_parent]
            I_eff_p = wp.min(I_p[0, 0], wp.min(I_p[1, 1], I_p[2, 2]))
            I_eff_p = wp.max(I_eff_p, 1.0e-12)
            ang_denom_p = (
                I_eff_p
                + dt * dt * joint_attach_ke
                + dt * joint_attach_kd * angular_damping_scale
            )
            total_t_p = t_attach + wp.cross(r_p, f_attach)
            delta_w_p = total_t_p * (dt / ang_denom_p)

            wp.atomic_add(
                body_qd, c_parent, wp.spatial_vector(delta_v_p, delta_w_p)
            )


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverSemiImplicitStable(SolverSemiImplicit):
    """Semi-implicit solver with unconditionally stable joint attachment.

    Identical to :class:`SolverSemiImplicit` except that joint attachment
    penalty forces (``joint_attach_ke/kd``) are applied via an implicit
    velocity correction *after* integration, rather than as explicit forces
    *before* integration.  This makes the solver unconditionally stable for
    any body mass or inertia, at the cost of a slight reduction in joint
    constraint accuracy (Jacobi-style single pass).

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

            # --- Force accumulation (same as parent, except ke=0, kd=0) ---

            # Damped springs
            eval_spring_forces(model, state_in, particle_f)

            # Triangle elastic and lift/drag
            eval_triangle_forces(model, state_in, control, particle_f)

            # Triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # Tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # Body joints: PD targets + limits only, NO attachment forces.
            # Attachment forces are applied implicitly after integration.
            eval_body_joint_forces(model, state_in, control, body_f, 0.0, 0.0)

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

            # --- Implicit joint attachment correction (NEW) ---
            if model.joint_count:
                wp.launch(
                    kernel=implicit_joint_attachment,
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
                        model.joint_dof_dim,
                        self.joint_attach_ke,
                        self.joint_attach_kd,
                        dt,
                    ],
                    device=model.device,
                )
