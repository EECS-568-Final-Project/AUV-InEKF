import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from common import *
from readData import process_sensor_data


def plot_xyz(
    data: list[tuple[float, Vec3]],
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    title: str = "Sensor Data",
) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)
    times = [time for time, _ in data]

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(x_label)
    axs[0].plot(times, [vec.x for _, vec in data])
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(y_label)
    axs[1].plot(times, [vec.y for _, vec in data])

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel(z_label)
    axs[2].plot(times, [vec.z for _, vec in data])

    for ax, label in zip(axs, (x_label, y_label, z_label)):
        ax.set_title(f"{label} data")
        ax.grid()
    plt.show()


def plot_imu(data: list[SensorData]) -> None:
    plot_xyz(
        [(e.time, e.lin_acc) for e in data if e.lin_acc is not None],
        title="Linear acceleration",
    )
    plot_xyz(
        [(e.time, e.ang_vel) for e in data if e.ang_vel is not None],
        title="Angular velocity",
    )


def plot_dvl(data: list[SensorData]) -> None:
    plot_xyz([(e.time, e.dvl) for e in data if e.dvl is not None], title="Dvl velocity")


def fillInGaps(sensor_data):
    for sensor in sensor_data:
        for idx, value in enumerate(sensor):
            if value is not None:
                firstIndex = value
                break
            elif idx == 0 and value:
                break

        sensor[:idx] = [firstIndex] * idx


def formatSensorData(sensor_data: list[SensorData]):
    def appendLast(dataSources):
        for sensor in dataSources:
            sensor.append(sensor[-1])

    imu, dvl, depth, ahrs = [], [], [], []
    lastValues = sensor_data[0]

    imu.append((lastValues.lin_acc, lastValues.ang_vel, lastValues.imu_dt))
    dvl.append(lastValues.dvl)
    depth.append(lastValues.depth)
    ahrs.append(lastValues.mag)

    for data in sensor_data[1:]:
        if data.lin_acc is not None:
            imu.append((data.lin_acc, data.ang_vel, data.imu_dt))
            appendLast([dvl, depth, ahrs])

        elif data.dvl is not None:
            dvl.append(data.dvl)
            appendLast([imu, depth, ahrs])

        elif data.depth is not None:
            depth.append(data.depth)
            appendLast([imu, dvl, ahrs])

        elif data.mag is not None:
            ahrs.append(data.mag)
            appendLast([imu, dvl, depth])

        lastValues = data

    fillInGaps([imu, dvl, depth, ahrs])

    return imu, dvl, depth, ahrs


def plotRobotData(
    sensor_data: list[SensorData], predicted_states: list[np.ndarray], noAHRS_predictedStates: list[np.ndarray] = []
) -> None:
    """
    Compare raw SensorData to your list of SE(3) predicted_states.

    sensor_data : list of SensorData
        .time   float
        .dvl    Vec3 (velocity in body frame)
        .depth  float (meters)
        .mag    Vec3 (magnetometer vector, body frame)
        … (lin_acc, ang_vel, imu_dt are available if you want to plot them)

    predicted_states : list of 5x5 np.ndarray
        [ R (3x3) | v (3x1) | p (3x1) ]
        [   0     |   1     |   0     ]
        [   0     |   0     |   1     ]
    """
    imu, dvl, depth, ahrs = formatSensorData(sensor_data)

    ##### FIXME
    size = len(predicted_states) * 0.2
    imu = imu[: int(size)]
    dvl = dvl[: int(size)]
    depth = depth[: int(size)]
    ahrs = ahrs[: int(size)]
    predicted_states = predicted_states[: int(size)]
    noAHRS_predictedStates = noAHRS_predictedStates[: int(size)]
    #####

    # --- extract timestamps ---
    t = np.array([sd.time for sd in sensor_data])
    t = t[: len(predicted_states)]

    # --- 1) velocities: DVL vs predicted ---
    try: 
        pred_vel = np.vstack([S[:3, 3] for S in predicted_states])
        sensor_vel = np.vstack([[data.x, data.y, data.z] for data in dvl])
        noAHRS_pred_vel = np.vstack([S[:3, 3] for S in noAHRS_predictedStates])

        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, comp in enumerate(["x", "y", "z"]):
            ax[i].plot(t, sensor_vel[:, i], label=f"DVL vel {comp}")
            ax[i].plot(t, pred_vel[:, i], "--", label=f"pred vel {comp}", color="orange")
            ax[i].plot(t, noAHRS_pred_vel[:, i], ":", label=f"pred vel {comp} (no AHRS)", color='red')

            ax[i].set_ylabel(f"v_{comp} (m/s)")
            ax[i].legend(loc="best", fontsize="small")
        ax[-1].set_xlabel("time (s)")
        fig.suptitle("Linear Velocity Comparison")
    except Exception as e:
        print("No DVL data available to plot.")
        print(e)

    # --- 2) depth vs predicted z-position ---
    try:
        pred_pos = np.vstack([S[:3, 4] for S in predicted_states])
        noAHRS_pred_pos = np.vstack([S[:3, 4] for S in noAHRS_predictedStates])
        sensor_depth = np.array([data for data in depth])

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(t, sensor_depth, label="depth sensor")
        ax2.plot(t, pred_pos[:, 2], "--", label="predicted z", color="orange")
        ax2.plot(t, noAHRS_pred_pos[:, 2], ":", label="predicted z (no AHRS)", color='red')

        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("depth (m)")
        ax2.legend()
        ax2.set_title("Depth vs Predicted Z")
    except Exception as e:
        print("No depth data available to plot.")
        print(e)

    # --- 3) orientation: predicted roll, pitch, yaw ---
    try:
        # 1. Extract predicted Euler angles (ZYX = yaw, pitch, roll)
        Rs = np.array([S[:3, :3] for S in predicted_states])
        pred_euler = R.from_matrix(Rs).as_euler('zyx', degrees=True)

        noAHRS_Rs = np.array([S[:3, :3] for S in noAHRS_predictedStates])
        noAHRS_pred_euler = R.from_matrix(noAHRS_Rs).as_euler('zyx', degrees=True)

        # 2. Extract raw magnetometer vector components
        mag_x = np.degrees([m.x for m in ahrs])
        mag_y = np.degrees([m.y for m in ahrs])
        mag_z = np.degrees([m.z for m in ahrs])

        # 3. Plot
        labels = ["yaw", "pitch", "roll"]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(t, mag_z, label="Sensor Reading", color="blue")
        axs[0].plot(t, pred_euler[:, 0], '--', label="predicted yaw", color="orange")
        axs[0].plot(t, noAHRS_pred_euler[:, 0], ':', label="predicted yaw (no AHRS)", color="red")

        axs[1].plot(t, mag_y, label="Sensor Reading", color="blue")
        axs[1].plot(t, pred_euler[:, 1], '--', label="predicted pitch", color="orange")
        axs[1].plot(t, noAHRS_pred_euler[:, 1], ':', label="predicted pitch (no AHRS)", color="red")

        axs[2].plot(t, mag_x, label="Sensor Reading", color="blue")
        axs[2].plot(t, pred_euler[:, 2], '--', label="predicted roll", color="orange")
        axs[2].plot(t, noAHRS_pred_euler[:, 2], ':', label="predicted roll (no AHRS)", color="red")

        for i, label in enumerate(labels):
            axs[i].set_ylabel(f"{label} (deg)")
            axs[i].legend()
            axs[i].grid(True)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle("Predicted Roll, Pitch, Yaw")
    except Exception as e:
        print("Orientation plot failed:", e)

    # --- 4) (Optional) 3D trajectory plot ---
    try:
        fig4 = plt.figure(figsize=(6, 6))
        ax4 = fig4.add_subplot(projection="3d")
        ax4.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], "--", label="predicted", color="orange")
        ax4.plot(noAHRS_pred_pos[:, 0], noAHRS_pred_pos[:, 1], noAHRS_pred_pos[:, 2], ":", label="predicted (no AHRS)", color='red')

        # if you have an external “true” position series, plot it here too
        ax4.set_xlabel("X (m)")
        ax4.set_ylabel("Y (m)")
        ax4.set_zlabel("Z (m)")
        ax4.set_title("3D Predicted Trajectory")
        ax4.legend()
    except Exception as e:
        print("No 3D trajectory data available to plot.")
        print(e)

    plt.show()


def main():
    # Read the data from the file
    data = process_sensor_data("stationary")
    plot_imu(data)
    plot_dvl(data)


if __name__ == "__main__":
    main()
