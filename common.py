from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import NamedTuple


type FloatMat = npt.NDArray[np.floating]


class Vec3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def __mul__(self, other: float) -> "Vec3":
        return Vec3(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __rmul__(self, other: float) -> "Vec3":
        return self * other

    def __truediv__(self, other: float) -> "Vec3":
        return self * (1 / other)

    @staticmethod
    def from_matrix(matrix: FloatMat) -> "Vec3":
        return Vec3(matrix[0], matrix[1], matrix[2])

    def as_matrix(self) -> FloatMat:
        return np.array([self.x, self.y, self.z])


@dataclass
class SensorData:
    time: float
    imu_dt: float | None
    dvl: Vec3 | None
    lin_acc: Vec3 | None
    ang_vel: Vec3 | None
    depth: float | None
    mag: Vec3 | None
