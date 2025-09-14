import numpy as np
from casadi import *

class MPC:
    def __init__(self, dt=0.1, N=20, d_des=10.0, v_ref=25.0):
        self.dt = dt
        self.N = N
        self.d_des = d_des
        self.v_ref = v_ref

        # Weights
        self.q_d = 10.0
        self.q_v = 1.0
        self.r   = 0.1
        self.q_slack = 1e4  # big penalty for soft-constraint violations

        # Limits
        self.a_min = -3.0
        self.a_max =  2.0
        self.v_min =  0.0
        self.v_max = 40.0
        self.d_min =  5.0

        # Dynamics: x = [gap, v_ego], u = accel, parameter p = v_lead
        x = MX.sym('x', 2)
        u = MX.sym('u')
        p = MX.sym('p')  # lead speed
        x_next = vertcat(
            x[0] + self.dt * (p - x[1]),   # gap_{k+1} = gap_k + dt*(v_lead - v_ego)
            x[1] + self.dt * u             # v_{k+1} = v_k + dt*a
        )
        self.f = Function('f', [x, u, p], [x_next])

    def predict(self, obs, deterministic=True):
        # Unnormalize obs
        ego_speed  = float(obs[0]) * 30.0
        gap        = float(obs[1]) * 150.0
        lead_speed = float(obs[2]) * 30.0

        x0 = np.array([gap, ego_speed], dtype=float)

        opti = Opti()
        X = opti.variable(2, self.N + 1)      # states
        U = opti.variable(1, self.N)          # accel inputs
        p_vlead = opti.parameter()            # parameter: constant lead speed over horizon

        # Soft constraint slacks (>=0)
        S_gap  = opti.variable(1, self.N)     # for gap lower bound
        S_vlo  = opti.variable(1, self.N)     # for v_min
        S_vhi  = opti.variable(1, self.N)     # for v_max
        opti.subject_to(S_gap  >= 0)
        opti.subject_to(S_vlo  >= 0)
        opti.subject_to(S_vhi  >= 0)

        # Initial condition
        opti.subject_to(X[:, 0] == x0)

        cost = 0
        for k in range(self.N):
            # Track desired gap and match speed to min(v_ref, v_lead)
            v_target = min(self.v_ref, lead_speed)
            gap_err = X[0, k] - self.d_des
            vel_err = X[1, k] - v_target

            cost += self.q_d * gap_err**2 + self.q_v * vel_err**2 + self.r * U[0, k]**2
            cost += self.q_slack * (S_gap[0, k] + S_vlo[0, k] + S_vhi[0, k])

            # Dynamics
            opti.subject_to(X[:, k+1] == self.f(X[:, k], U[0, k], p_vlead))

            # Input bounds
            opti.subject_to(opti.bounded(self.a_min, U[0, k], self.a_max))

            # **State bounds from k=1 onward only** (k=0 is fixed to possibly-invalid x0)
            # gap >= d_min - S_gap, v in [v_min - S_vlo, v_max + S_vhi]
            opti.subject_to(X[0, k] >= self.d_min - S_gap[0, k])
            opti.subject_to(X[1, k] >= self.v_min - S_vlo[0, k])
            opti.subject_to(X[1, k] <= self.v_max + S_vhi[0, k])

        # Optional: small terminal cost on final state (encourages convergence)
        cost += 0.1 * (X[0, self.N] - self.d_des)**2 + 0.01 * (X[1, self.N] - min(self.v_ref, lead_speed))**2

        opti.minimize(cost)

        # IPOPT settings + good initials
        opti.set_value(p_vlead, lead_speed)
        opti.set_initial(X, np.tile(x0.reshape(2,1), (1, self.N+1)))
        opti.set_initial(U, 0.0)
        opti.set_initial(S_gap, 0.0)
        opti.set_initial(S_vlo, 0.0)
        opti.set_initial(S_vhi, 0.0)

        opti.solver('ipopt', {
            "print_time": 0,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-6
        })

        try:
            sol = opti.solve()
            a_opt = float(sol.value(U[0, 0]))
            v_next = float(sol.value(X[1, 1]))  # next-step target speed
        except RuntimeError:
            print("[MPC] Solver failed, returning 0.")
            a_opt = 0.0
            v_next = ego_speed

        return np.array([a_opt], dtype=float), v_next
