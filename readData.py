from pprint import pprint
import csv
from common import *


def read_csv_data(csvFilePath, dataSource="IMU") -> list[SensorData]:
    sensor_data_list = []

    try:
        with open(csvFilePath, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                sensor_data = SensorData(
                    time=float(row["Time"]),
                    imu_dt=None,
                    dvl=None,
                    lin_acc=None,
                    ang_vel=None,
                    depth=None,
                    mag=None,
                )

                if dataSource == "IMU":
                    linear_acceleration = Vec3(
                        float(row["dvel.x"]), float(row["dvel.y"]), float(row["dvel.z"])
                    )

                    angular_velocity = Vec3(
                        float(row["dtheta.x"]),
                        float(row["dtheta.y"]),
                        float(row["dtheta.z"]),
                    )

                    sensor_data.lin_acc = linear_acceleration
                    sensor_data.ang_vel = angular_velocity
                    sensor_data.imu_dt = float(row["dt"])

                elif dataSource == "DVL":
                    velocity = Vec3(
                        float(row["velocityA"]),
                        float(row["velocityB"]),
                        float(row["velocityC"]),
                    )
                    sensor_data.dvl = velocity

                elif dataSource == "Depth":
                    sensor_data.depth = float(row["data"])

                elif dataSource == "AHRS":
                    magnotometer_data = Vec3(
                        float(row["theta_0"]),
                        float(row["theta_1"]),
                        float(row["theta_2"]),
                    )

                    sensor_data.mag = magnotometer_data

                sensor_data_list.append(sensor_data)

        return sensor_data_list

    except Exception as e:
        print(f"Error opening file: {e}")
        return []


def aggregate_results(results: list[list[SensorData]]) -> list[SensorData]:
    aggregated_data = []
    indexes = [0] * len(results)
    while True:
        min_time = float("inf")
        min_index = -1

        for i, data in enumerate(results):
            if indexes[i] < len(data) and data[indexes[i]].time < min_time:
                min_time = data[indexes[i]].time
                min_index = i

        if min_index == -1:
            break

        # DataType[min_index] [currentIndex[min_index]]
        aggregated_data.append(results[min_index][indexes[min_index]])
        indexes[min_index] += 1

    return aggregated_data


def process_sensor_data(name="stationary") -> list[SensorData]:
    DIR = "data/"
    dataSources = [
        ("_imu.csv", "IMU"),
        ("_dvl.csv", "DVL"),
        ("_depth.csv", "Depth"),
        ("_ahrs.csv", "AHRS"),
    ]

    results: list[list[SensorData]] = []
    for dataSource in dataSources:
        filePath = str(DIR + name + dataSource[0])
        sensor_data = read_csv_data(filePath, dataSource[1])
        results.append(sensor_data)

    aggregated = aggregate_results(results)
    for datum in aggregated:
        if datum.imu_dt is not None:
            assert datum.lin_acc is not None
            assert datum.ang_vel is not None
            datum.lin_acc /= datum.imu_dt
            datum.ang_vel /= datum.imu_dt
    return aggregated


def main():
    process_sensor_data("stationary")


if __name__ == "__main__":
    main()
