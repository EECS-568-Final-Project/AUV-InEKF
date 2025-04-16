import numpy as np

from common import *
import readData
from IEKF import IEKF

def run_filter(sensor_data_list: list[SensorData], iekf: IEKF):
    predictedStates = [iekf.state.copy()]
    timestamps = [sensor_data_list[0].time]

    for sensor_data in sensor_data_list[1:]:
        if sensor_data.lin_acc is not None:
            control = {
                'linear_acceleration': sensor_data.lin_acc,
                'angular_velocity': sensor_data.ang_vel,
            }
            iekf.predict(control, sensor_data.imu_dt)
        if sensor_data.depth is not None:
            iekf.update_depth(sensor_data.depth)
        if sensor_data.dvl is not None:
            iekf.update_dvl(sensor_data.dvl)


        timestamps.append(sensor_data.time)
        predictedStates.append(iekf.state.copy())


def main():

    sensor_data = readData.process_sensor_data("stationary")

    '''
        - Initial State
        - Initial Covariance
        - Process Noise
        - Measurement Noise
    '''
    I_pose = np.eye(5)
    covariance = np.eye(15)

    process_noise = np.eye(15)
    depth_measurement_noise = np.zeros((3, 3))
    depth_measurement_noise[2, 2] = 1 # Depth measurement noise
    dvl_measurement_noise = np.eye(3)
    ahrs_measurement_noise = np.eye(3)

    measurement_noise = {
        'depth': depth_measurement_noise,
        'dvl': dvl_measurement_noise,
        'ahrs': ahrs_measurement_noise
    }

    iekf = IEKF(
        initial_state=I_pose,
        initial_covariance=covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )

    run_filter(sensor_data, iekf)

if __name__ == '__main__':
    main()