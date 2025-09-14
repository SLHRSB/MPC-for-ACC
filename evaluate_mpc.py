# evaluate_mpc.py

import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from PS_env_helper import Reciver_UDP2, Transmitter_UDP_2
from MPC import MPC
from run_mpc_loop import parse_sensor_data, accel_to_actuators, reset_sim

# ---------- Config ----------
NUM_EPISODES = 100
GOAL_DISTANCE_M = 200.0
MAX_STEPS_PER_EP = 250
RESET_COOLDOWN_SEC = 0.3
RESET_DESIRED_V = 10.0
LOG_DIR = "logs/MPC_Eval/"
EXPERIMENT_NAME = "MPC_ACC_Test"
# ----------------------------


def reward_step(v_ego):
    """Step reward shaped like in PPO env."""
    reward = -0.05
    max_reward_speed = 1.0
    if v_ego <= 5:
        reward += -max_reward_speed * v_ego / 10.0
    elif v_ego <= 25:
        reward += max_reward_speed * v_ego / 10.0
    else:
        reward += -max_reward_speed * v_ego / 10.0
    return reward


def evaluate_mpc():
    # Init UDP I/O
    sensor_udp = Reciver_UDP2("env_info_udp", 8031)
    sensor_udp.build()
    actuator_udp = Transmitter_UDP_2("send_action", 8032)

    # Init MPC
    mpc = MPC()
    dt = float(mpc.dt)

    # TensorBoard writer
    timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
    log_path = os.path.join(LOG_DIR, EXPERIMENT_NAME + "_" + timestamp)
    writer = SummaryWriter(log_path)

    # Metrics
    total_collisions = 0
    episode_rewards = []
    episode_speeds = []

    try:
        for ep in range(1, NUM_EPISODES + 1):
            # Reset env
            reset_sim(actuator_udp, desired_velocity=RESET_DESIRED_V, wait_s=0.1)
            time.sleep(RESET_COOLDOWN_SEC)

            done = False
            step = 0
            ep_speed = []
            ep_reward = 0.0
            status = None

            next_t = time.time()

            while not done:
                # Get sensor data
                raw_data = sensor_udp.get()
                obs, v_ego, gap, v_lead, x_host, has_collision = parse_sensor_data(raw_data)

                # Episode termination checks
                if x_host > GOAL_DISTANCE_M:
                    status = "End of the road!"
                    ep_reward += 5
                    done = True
                elif has_collision:
                    status = "Collision!"
                    total_collisions += 1
                    ep_reward += -10
                    done = True
                elif step >= MAX_STEPS_PER_EP:
                    status = "Took Too Long!"
                    ep_reward += -10
                    done = True

                if done:
                    break

                # Run MPC
                action, v_next = mpc.predict(obs)
                accel = float(action[0])
                throttle, brake = accel_to_actuators(accel, mpc.a_min, mpc.a_max)

                # Send command
                actuator_udp.send_data(0.0, float(v_next), throttle, brake, 0)

                # Step metrics
                ep_speed.append(v_ego)
                ep_reward += reward_step(v_ego)

                # Timing
                next_t += dt
                sleep_time = next_t - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_t = time.time()

                step += 1

            # Episode metrics
            mean_speed = np.mean(ep_speed) if ep_speed else 0.0
            collision_rate = total_collisions / ep if ep > 0 else 0.0

            episode_rewards.append(ep_reward)
            episode_speeds.append(mean_speed)

            writer.add_scalar("MPC/Episode Reward", ep_reward, ep)
            writer.add_scalar("MPC/Mean Speed", mean_speed, ep)
            writer.add_scalar("MPC/Collision Rate", collision_rate, ep)

            print(f"[MPC] Episode {ep:03d}: {status}, steps={step}, "
                  f"mean_speed={mean_speed:.2f}, reward={ep_reward:.2f}, "
                  f"coll_rate={collision_rate:.2f}")

        # Final summary
        mean_reward = np.mean(episode_rewards)
        mean_speed = np.mean(episode_speeds)
        print("===================================")
        print(f"[MPC] Finished {NUM_EPISODES} episodes")
        print(f"Mean reward: {mean_reward:.2f}")
        print(f"Mean speed: {mean_speed:.2f}")
        print(f"Collision rate: {total_collisions/NUM_EPISODES:.2f}")
        print("===================================")

    finally:
        writer.close()
        try:
            sensor_udp.close()
            actuator_udp.close()
        except Exception:
            pass


if __name__ == "__main__":
    evaluate_mpc()
