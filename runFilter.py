import numpy as np

from common import *
import readData
from IEKF import IEKF, ControlInput, SensorNoise

from pprint import pprint
from tqdm import tqdm

from plotData import plotRobotData


def run_filter(sensor_data_list: list[SensorData], iekf: IEKF):
    predictedStates = [iekf.state.copy()]
    timestamps = [sensor_data_list[0].time]

    for sensor_data in tqdm(sensor_data_list[1:]):
        # breakpoint()
        if sensor_data.imu_dt is not None:
            assert sensor_data.lin_acc is not None
            assert sensor_data.ang_vel is not None
            control: ControlInput = {
                "linear_acceleration": sensor_data.lin_acc,
                "angular_velocity": sensor_data.ang_vel,
            }
            iekf.predict(control, sensor_data.imu_dt)
        if sensor_data.depth is not None:
            iekf.update_depth(sensor_data.depth)
        if sensor_data.dvl is not None:
            iekf.update_dvl(sensor_data.dvl)
        # if sensor_data.mag is not None:
        #     iekf.update_ahrs(sensor_data.mag)

        timestamps.append(sensor_data.time)
        predictedStates.append(iekf.state.copy())

    return predictedStates, timestamps


zero = Vec3(0.0, 0.0, 0.0)


def set_data_zero(data: SensorData):
    if data.dvl is not None:
        data.dvl = zero
    if data.lin_acc is not None:
        data.lin_acc = Vec3(0.0, 0.0, 9.81)
    if data.ang_vel is not None:
        data.ang_vel = zero
    if data.depth is not None:
        data.depth = 0.0
    if data.mag is not None:
        data.mag = zero


def main():

    print("Reading Data...")
    sensor_data = readData.process_sensor_data("stationary")
    t0 = sensor_data[0].time
    for data in sensor_data:
        # set_data_zero(data)
        data.time -= t0
    print("Data Read")

    print("Initializing Filter...")
    """
        - Initial State
        - Initial Covariance
        - Process Noise
        - Measurement Noise
    """
    I_pose = np.eye(5)
    covariance = np.eye(15)

    process_noise = np.zeros((15, 15))
    process_noise[0:3, 0:3] = np.eye(3) * 1  # Rotation noise
    process_noise[3:6, 3:6] = np.eye(3) * 5  # Velocity noise
    process_noise[6:9, 6:9] = np.eye(3) * 1  # Position noise
    process_noise[9:12, 9:12] = np.eye(3) * 0.1  # Bias noise
    process_noise[12:15, 12:15] = np.eye(3) * 1  # Bias noise

    depth_measurement_noise = np.zeros((3, 3))
    depth_measurement_noise[2, 2] = 0.05  # Depth measurement noise

    dvl_measurement_noise = np.eye(3) * 0.5
    ahrs_measurement_noise = np.eye(3) * 0.02

    measurement_noise: SensorNoise = {
        "depth": depth_measurement_noise,
        "dvl": dvl_measurement_noise,
        "ahrs": ahrs_measurement_noise,
    }

    iekf = IEKF(
        initial_state=I_pose,
        initial_covariance=covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    print("Running Filter...")
    predictedStates, timestamps = run_filter(sensor_data, iekf)

    print("Plotting Results")
    plotRobotData(sensor_data, predictedStates)


if __name__ == "__main__":
    main()
