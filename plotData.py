import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from common import *


def readIMUData(file_name: str) -> tuple[list[Vec3], list[Vec3]]:
    data = {
        'posisiton': [],
        'velocity': [],
    }

    with open(file_name, 'r') as file:
        # file is csv format first column is time, second is position.x, third is position.y, fourth is position.z, fith is velocity.x, sixth is velocity.y, seventh is velocity.z
        for line in file:
            # Skip first line
            if line.startswith("Time"):
                continue

            line = line.strip().split(',')
            if len(line) < 7:
                continue

            data['posisiton'].append(Vec3(float(line[1]), float(line[2]), float(line[3])))
            data['velocity'].append(Vec3(float(line[4]), float(line[5]), float(line[6])))
    return data['posisiton'], data['velocity']


def plotPath(data: list[Vec3]) -> None:
    # Plot each axis separately
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Sensor Data")
    fig.subplots_adjust(hspace=0.5)
    labels = ["X", "Y", "Z"]

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X")
    axs[0].plot(
        [pose.x for pose in data],
    )
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Y")
    axs[1].plot(
        [pose.y for pose in data],
    )

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Z")
    axs[2].plot(
        [pose.z for pose in data],
    )
    for i, ax in enumerate(axs):
        ax.set_title(f"Path in {labels[i]} direction")
        ax.grid()
    plt.show()




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


def plotRobotData(sensor_data: list[SensorData],
                  predicted_states: list[np.ndarray]) -> None:
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
    size = len(predicted_states) * 1
    imu = imu[:int(size)]
    dvl = dvl[:int(size)]
    depth = depth[:int(size)]
    ahrs = ahrs[:int(size)]
    predicted_states = predicted_states[:int(size)]
    #####

    # --- extract timestamps ---
    t = np.array([sd.time for sd in sensor_data])
    t = t[:len(predicted_states)]

    # --- 1) velocities: DVL vs predicted ---
    sensor_vel = np.vstack([ [data.x, data.y, data.z] for data in dvl ])
    pred_vel   = np.vstack([ S[:3, 3] for S in predicted_states ])

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, comp in enumerate(['x','y','z']):
        ax[i].plot(t, sensor_vel[:,i],    label=f'DVL vel {comp}')
        ax[i].plot(t, pred_vel[:,i], '--', label=f'pred vel {comp}')
        ax[i].set_ylabel(f'v_{comp} (m/s)')
        ax[i].legend(loc='best', fontsize='small')
    ax[-1].set_xlabel('time (s)')
    fig.suptitle('Linear Velocity Comparison')

    # --- 2) depth vs predicted z-position ---
    sensor_depth = np.array([data for data in depth])
    pred_pos      = np.vstack([ S[:3, 4] for S in predicted_states ])


    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(t, sensor_depth,    label='depth sensor')
    ax2.plot(t, pred_pos[:,2], '--', label='predicted z')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('depth (m)')
    ax2.legend()
    ax2.set_title('Depth vs Predicted Z')

    # --- 3) heading: magnetometer vs predicted yaw ---
    # magnetometer-based heading in degrees
    sensor_yaw = np.degrees([
        np.arctan2(data.y, data.x)
        for data in ahrs
    ])
    # predicted orientation → extract yaw from R
    # Rs = np.array([ S[:3,:3] for S in predicted_states ])
    # pred_euler = R.from_matrix(Rs).as_euler('xyz', degrees=True)  # roll,pitch,yaw
    # pred_yaw   = pred_euler[:,2]

    # fig3, ax3 = plt.subplots(figsize=(8,4))
    # ax3.plot(t, sensor_yaw,    label='mag heading')
    # ax3.plot(t, pred_yaw,   '--', label='predicted yaw')
    # ax3.set_xlabel('time (s)')
    # ax3.set_ylabel('yaw (deg)')
    # ax3.legend()
    # ax3.set_title('Heading Comparison')

    # --- 4) (Optional) 3D trajectory plot ---
    fig4 = plt.figure(figsize=(6,6))
    ax4 = fig4.add_subplot(projection='3d')
    ax4.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2], '--', label='predicted')
    # if you have an external “true” position series, plot it here too
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)'); ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Predicted Trajectory')
    ax4.legend()

    plt.show()

'''
def plotRobotPoses(sensor_data: list[SensorData],
                   predicted_states: list[np.ndarray],
                   timestamps: list[float]) -> None:
    """
    Plot the sensor data and predicted states from the InEKF.
    
    Args:
        sensor_data: List of SensorData objects containing raw measurements
        predicted_states: List of SE(3) matrices representing predicted robot states
        timestamps: List of timestamps for each measurement/prediction
    """
    # Extract sensor measurements
    dvl_measurements = [data.dvl for data in sensor_data]
    depth_measurements = [data.depth for data in sensor_data]
    mag_measurements = [data.mag for data in sensor_data]
    
    # Extract predicted positions, velocities, and rotations from the state matrices
    pred_positions = []
    pred_velocities = []
    pred_rotations = []
    
    for state in predicted_states:
        # Extract position (top-right 3x1 block)
        position = state[:3, 4]
        pred_positions.append(position)
        
        # Extract velocity (middle column of the upper block)
        velocity = state[:3, 3]
        pred_velocities.append(velocity)
        
        # Extract rotation matrix (top-left 3x3 block)
        rotation = state[:3, :3]
        # Convert rotation matrix to euler angles for easier visualization
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pred_rotations.append(np.array([roll, pitch, yaw]))
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 15))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(321, projection='3d')
    
    # Plot predicted trajectory
    pred_x = [pos[0] for pos in pred_positions]
    pred_y = [pos[1] for pos in pred_positions]
    pred_z = [pos[2] for pos in pred_positions]
    ax1.plot(pred_x, pred_y, pred_z, 'r-', linewidth=2, label='Predicted Path')
    
    # Add start and end markers
    if len(pred_positions) > 0:
        ax1.plot([pred_x[0]], [pred_y[0]], [pred_z[0]], 'go', markersize=8, label='Start')
        ax1.plot([pred_x[-1]], [pred_y[-1]], [pred_z[-1]], 'bo', markersize=8, label='End')
    
    ax1.set_title('3D Predicted Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.grid(True)
    
    # Position plot
    ax2 = fig.add_subplot(322)
    ax2.plot(timestamps, [pos[0] for pos in pred_positions], 'r-', label='X')
    ax2.plot(timestamps, [pos[1] for pos in pred_positions], 'g-', label='Y')
    ax2.plot(timestamps, [pos[2] for pos in pred_positions], 'b-', label='Z')
    ax2.set_title('Position vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.grid(True)
    
    # Velocity plot
    ax3 = fig.add_subplot(323)
    ax3.plot(timestamps, [vel[0] for vel in pred_velocities], 'r-', label='X')
    ax3.plot(timestamps, [vel[1] for vel in pred_velocities], 'g-', label='Y')
    ax3.plot(timestamps, [vel[2] for vel in pred_velocities], 'b-', label='Z')
    ax3.set_title('Predicted Velocity vs Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend()
    ax3.grid(True)
    
    # DVL measurements plot
    ax4 = fig.add_subplot(324)
    ax4.plot(timestamps, [dvl.x for dvl in dvl_measurements], 'r--', label='X')
    ax4.plot(timestamps, [dvl.y for dvl in dvl_measurements], 'g--', label='Y')
    ax4.plot(timestamps, [dvl.z for dvl in dvl_measurements], 'b--', label='Z')
    ax4.set_title('DVL Measurements vs Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.legend()
    ax4.grid(True)
    
    # Orientation plot (Euler angles)
    ax5 = fig.add_subplot(325)
    ax5.plot(timestamps, [rot[0] for rot in pred_rotations], 'r-', label='Roll')
    ax5.plot(timestamps, [rot[1] for rot in pred_rotations], 'g-', label='Pitch')
    ax5.plot(timestamps, [rot[2] for rot in pred_rotations], 'b-', label='Yaw')
    ax5.set_title('Orientation vs Time')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (rad)')
    ax5.legend()
    ax5.grid(True)
    
    # Depth measurements plot
    ax6 = fig.add_subplot(326)
    ax6.plot(timestamps, depth_measurements, 'b-', label='Depth Measurement')
    ax6.plot(timestamps, [pos[2] for pos in pred_positions], 'r--', label='Predicted Z')
    ax6.set_title('Depth vs Time')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Depth (m)')
    ax6.legend()
    ax6.grid(True)
    ax6.invert_yaxis()  # Depth typically increases downward
    
    plt.tight_layout()
    plt.show()
    
    return

'''

def main():
    # Read the data from the file
    position, velocity = readIMUData("data/forward_imu.csv")
    print(f"Position: {position}")
    # Plot the path
    plotPath(position)
    plotPath(velocity)

if __name__ == "__main__":
    main()
