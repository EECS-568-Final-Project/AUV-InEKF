import numpy as np
from scipy.linalg import expm, logm
from common import *


class IEKF:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.X = initial_state
        self.P = initial_covariance

        self.Q = process_noise
        self.R_depth = measurement_noise['depth']
        self.R_dvl = measurement_noise['dvl']
        self.R_ahrs = measurement_noise['ahrs']

        self.bias = np.zeros(3)

    def predict(self, control_input, dt):
        acceleration = control_input['linear_acceleration']
        gyroscope = control_input['angular_velocity']

        acceleration_np = np.array([acceleration.x, acceleration.y, acceleration.z])
        gyroscope_np = np.array([gyroscope.x, gyroscope.y, gyroscope.z])

        gravity = np.array([0, 0, -9.81])

        zeros = np.zeros((3, 3))
        I = np.eye(3)
        g_cross = self._skew(gravity)



        R = self.X[:3, :3]
        v = self.X[:3, 3]
        p = self.X[:3, 4]

        w_skew = self._skew(gyroscope_np)

        R_new = R @ expm(w_skew * dt)
        v_new = v + (R @ acceleration_np + gravity) * dt
        p_new = p + v * dt + 0.5 * (R @ acceleration_np + gravity) * dt ** 2



        X_new = np.zeros((5, 5))
        X_new[:3, :3] = R_new
        X_new[:3, 3] = v_new
        X_new[:3, 4] = p_new
        X_new[3, 3] = 1
        X_new[4, 4] = 1

        adj_X = np.block([
            [self._adjoint(X_new), np.zeros((9, 6))],
            [np.zeros((6, 9)), np.eye(6)]
        ])

        

        A = np.block([
            [zeros,     zeros,     zeros,         -R_new,     zeros],
            [g_cross,  zeros,     zeros,  -adj_X[3:6,0:3],  -R_new],
            [zeros,     I,        zeros,  -adj_X[6:9,0:3],   zeros],
            [zeros,     zeros,     zeros,          zeros,      zeros],
            [zeros,     zeros,     zeros,          zeros,      zeros]
        ])

        Phi = expm(A * dt)

        self.P = Phi @ (self.P + adj_X @ self.Q @ adj_X.T * dt) @ Phi.T

        self.X = X_new

        return self.X


    def update_depth(self, z):
        pass

    def update_dvl(self, z):
        pass

    def update_ahrs(self, z):
        pass

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    def _adjoint(self, X):
        R = X[:3, :3]
        p = X[:3, 4]

        adj = np.zeros((9, 9))

        adj[:3, :3] = R
        adj[3:6, 3:6] = R
        adj[6:9, 6:9] = R

        adj[3:6, :3] = self._skew(p) @ R

        return adj

    def run_filter(self, sensor_data):
        predicted_states = [self.X]
        
        timestamps = [sensor_data[0].timestamp]
        last_timestamp = sensor_data[0].timestamp

        for data in sensor_data[1:]:
            dt = data.timestamp - last_timestamp
            last_timestamp = data.timestamp

            # Predict
            control = {
                'linear_acceleration': data.linear_acceleration,
                'angular_velocity': data.angular_velocity,
            }
            self.predict(control, dt)

            # Update
            if data.depth is not None:
                self.update_depth(data.depth)
            if data.dvl is not None:
                self.update_dvl(data.dvl)
            if data.ahrs is not None:
                self.update_ahrs(data.ahrs)


            predicted_states.append(self.X.copy())
            timestamps.append(data.time)

        return predicted_states, timestamps



