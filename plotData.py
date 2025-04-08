import matplotlib.pyplot as plt
from Vec3 import Vec3


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

def main():
    # Read the data from the file
    position, velocity = readIMUData("data/forward_imu.csv")
    print(f"Position: {position}")
    # Plot the path
    plotPath(position)
    plotPath(velocity)

if __name__ == "__main__":
    main()
