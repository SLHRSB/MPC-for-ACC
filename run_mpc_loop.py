# run_mpc_loop.py

import time
import numpy as np

from PS_env_helper import Reciver_UDP2, Transmitter_UDP_2
from MPC import MPC


# ===================== Config =====================
GOAL_DISTANCE_M = 200.0      # success when ego x_pos exceeds this (meters)
MAX_STEPS_PER_EP = 250       # optional: end episode if too long (set None to disable)
RESET_DESIRED_V = 10.0       # m/s used during reset pulse
RESET_COOLDOWN_SEC = 0.3     # avoid immediate re-triggering after reset
# ==================================================


# ---------- Parsing helpers ----------

def _f(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)

def parse_sensor_data(data):
    """
    Extracts normalized obs for MPC + raw values for convenience.
    Expected keys in `data`:
      - "Vel": ego speed (m/s)
      - "R001": gap to lead (m)
      - "V001": lead speed (m/s)
      - "Pos": dict with "x" (ego x-position in meters)
      - "if_c": '1' if any collision detected, else '0'
      - "C_1", "C_2": strings with IDs of the two vehicles in collision
    """
    v_ego  = _f(data.get("Vel", 0.0))
    gap    = _f(data.get("R001", 150.0))
    v_lead = _f(data.get("V001", v_ego))
    if gap <= 0.0:
        gap = 150.0

    # Position for episode success check
    try:
        x_host = _f(data.get("Pos", {}).get("x", 0.0))
    except Exception:
        x_host = 0.0

    # Normalized obs for MPC
    iter_in_episod = 0  # placeholder to keep layout
    obs = np.array([
        np.clip(v_ego / 30.0,  0.0, 1.0),
        np.clip(gap   / 150.0, 0.0, 1.0),
        np.clip(v_lead/ 30.0,  0.0, 1.0),
        np.clip(iter_in_episod / 100.0, 0.0, 1.0)
    ], dtype=np.float32)

    # Collision detection like your env
    if_c = str(data.get("if_c", "0")).strip()
    c1   = str(data.get("C_1", "")).strip()
    c2   = str(data.get("C_2", "")).strip()
    collided = (if_c == '1') and (c1 == '11' or c2 == '11' or (c1 == '0' and c2 == '0'))

    return obs, v_ego, gap, v_lead, x_host, collided


def accel_to_actuators(accel, a_min, a_max):
    """
    Map acceleration (m/s^2) to throttle/brake in [0,1].
    Positive accel -> throttle, Negative accel -> brake.
    """
    accel = float(np.clip(accel, a_min, a_max))
    throttle = max(0.0, accel / a_max) if a_max > 0 else 0.0
    brake    = max(0.0, -accel / abs(a_min)) if a_min < 0 else 0.0
    return float(np.clip(throttle, 0.0, 1.0)), float(np.clip(brake, 0.0, 1.0))


# ---------- Reset helper ----------

def reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1):
    """
    Issues a reset pulse to Simulink exactly like your env.reset():
      1) send reset_flag = 1
      2) small wait
      3) send reset_flag = 0
    """
    offset = 0.0
    throttle = 0.0
    brake = 0.0
    actuator_udp.send_data(offset, desired_velocity, throttle, brake, 1)  # raise
    time.sleep(wait_s)
    actuator_udp.send_data(offset, desired_velocity, throttle, brake, 0)  # lower


# ---------- Main loop ----------

def main():
    # === Init UDP I/O ===
    sensor_udp = Reciver_UDP2("env_info_udp", 8031)
    sensor_udp.build()
    actuator_udp = Transmitter_UDP_2("send_action", 8032)

    # === Init MPC controller ===
    mpc = MPC()  # dt, N, limits inside MPC.py

    episode = 0
    step_in_episode = 0
    collision_count = 0

    # Cooldown to avoid immediate retrigger after a reset pulse
    reset_cooldown_until = 0.0

    print("Starting MPC control loop with REVEAL...")
    try:
        dt = float(mpc.dt)
        next_t = time.time()

        # (Optional) Hard-reset everything at start, so Simulink is in a known state
        reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1)
        reset_cooldown_until = time.time() + RESET_COOLDOWN_SEC
        episode += 1
        step_in_episode = 0
        print(f"[MPC] Episode {episode} started.")

        while True:
            # 1) Receive sensor data
            raw_data = sensor_udp.get()
            obs, v_ego, gap, v_lead, x_host, collided = parse_sensor_data(raw_data)

            # 2) Episode termination checks
            now = time.time()

            # 2a) Success if reached goal distance
            if x_host > GOAL_DISTANCE_M and now >= reset_cooldown_until:
                print(f"[MPC] SUCCESS: Episode {episode} reached {x_host:.1f} m (goal {GOAL_DISTANCE_M} m). Resetting...")
                reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1)
                reset_cooldown_until = time.time() + RESET_COOLDOWN_SEC
                # Start next episode
                episode += 1
                step_in_episode = 0
                print(f"[MPC] Episode {episode} started.")
                # Skip control for this tick; continue to next cycle
                next_t = time.time() + dt
                continue

            # 2b) Collision
            if collided and now >= reset_cooldown_until:
                collision_count += 1
                print(f"[MPC] COLLISION: Episode {episode} collided at x={x_host:.1f} m (total collisions {collision_count}). Resetting...")
                reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1)
                reset_cooldown_until = time.time() + RESET_COOLDOWN_SEC
                # Start next episode
                episode += 1
                step_in_episode = 0
                print(f"[MPC] Episode {episode} started.")
                next_t = time.time() + dt
                continue

            # 2c) Took too long (optional)
            if MAX_STEPS_PER_EP is not None and step_in_episode >= MAX_STEPS_PER_EP and now >= reset_cooldown_until:
                print(f"[MPC] TIMEOUT: Episode {episode} exceeded {MAX_STEPS_PER_EP} steps. Resetting...")
                reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1)
                reset_cooldown_until = time.time() + RESET_COOLDOWN_SEC
                # Start next episode
                episode += 1
                step_in_episode = 0
                print(f"[MPC] Episode {episode} started.")
                next_t = time.time() + dt
                continue

            # 3) MPC: get acceleration command and next-step target speed
            action, v_next = mpc.predict(obs)  # action: np.array([a_opt])
            accel = float(action[0])

            # 4) Convert acceleration to actuator commands
            throttle, brake = accel_to_actuators(accel, mpc.a_min, mpc.a_max)

            # 5) Send control command
            offset = 0.0
            desired_velocity = float(v_next)  # use MPC's predicted next-step speed
            actuator_udp.send_data(offset, desired_velocity, throttle, brake, 0)

            # 6) Bookkeeping
            step_in_episode += 1

            # 7) Timing to run close to MPC.dt
            next_t += dt
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're running behind, realign the schedule
                next_t = time.time()

    except KeyboardInterrupt:
        print("\n[MPC] Loop stopped by user.")
    except Exception as e:
        print(f"[MPC] Loop crashed: {e}")
        raise
    finally:
        # Best-effort close
        try:
            sensor_udp.close()
        except Exception:
            pass
        try:
            actuator_udp.close()
        except Exception:
            pass
        print("[MPC] I/O closed. Bye.")


if __name__ == "__main__":
    main()
