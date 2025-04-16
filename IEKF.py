import numpy as np
from scipy.linalg import expm, logm, block_diag
from numpy.linalg import inv
from common import *


class IEKF:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance

        self.process_noise = process_noise
        self.depth_measurement_noise = measurement_noise['depth']
        self.dvl_measurement_noise = measurement_noise['dvl']
        self.ahrs_noise = measurement_noise['ahrs']

        # Dict, {'linear_acceleration': Vec3, 'angular_velocity': Vec3}
        self.last_controlInput = None 

        self.bias = np.zeros(3)

    def predict(self, control_input, dt: float):
        self.last_controlInput = control_input

        acceleration_vec = control_input['linear_acceleration']
        gyroscope_vec = control_input['angular_velocity']

        acceleration_np = np.array([acceleration_vec.x, acceleration_vec.y, acceleration_vec.z])
        gyroscope_np = np.array([gyroscope_vec.x, gyroscope_vec.y, gyroscope_vec.z])

        gravity = np.array([0, 0, -9.81])

        zeros = np.zeros((3, 3))
        I = np.eye(3)
        g_cross = self._skew(gravity)

        w_skew = self._skew(gyroscope_np)

        pred_rotation = self.rotation @ expm(w_skew * dt)
        pred_velocity = self.velocity + (self.rotation @ acceleration_np + gravity) * dt
        pred_position = self.position + self.velocity * dt + 0.5 * (self.rotation @ acceleration_np + gravity) * (dt ** 2)


        new_state = np.zeros((5, 5))
        new_state[:3, :3] = pred_rotation
        new_state[:3, 3] = pred_velocity
        new_state[:3, 4] = pred_position
        new_state[3, 3] = 1
        new_state[4, 4] = 1

        adjoint = block_diag(self._adjoint(new_state), np.eye(6))

        error_dyanmics = np.block([
            [zeros,     zeros,     zeros,    -pred_rotation,          zeros],
            [g_cross,  zeros,     zeros,  -adjoint[3:6,0:3], -pred_rotation],
            [zeros,     I,        zeros,  -adjoint[6:9,0:3],          zeros],
            [zeros,     zeros,     zeros,          zeros,             zeros],
            [zeros,     zeros,     zeros,          zeros,             zeros],
        ])

        error_update = expm(error_dyanmics * dt)

        self.covariance = error_update @ (self.covariance + adjoint @ self.process_noise @ adjoint.T * dt) @ error_update.T

        self.state = new_state

        return self.state


    def update_depth(self, depth: float):
        zero = np.zeros((3, 3))
        measurement_jacobian = np.block([zero, zero, np.eye(3), zero, zero])

        I = np.eye(6)
        zero = np.zeros((9, 6))

        pred_measurement = measurement_jacobian @ block_diag(self._adjoint(self.inverse_state), np.eye(6))

        measurement = np.array([self.position[0], self.position[1], depth, 0, 1])

        # Make innovation vector
        V = (self.inverse_state @ measurement)[:3]

        # TODO: Figure out what is self.sys.invR from paper code
        # inverseNoise = np.zeros((3,3))
        # inverseNoise[-1, -1] = 1 / self.depth_measurement_noise
        
        pred_meas_cov = inv(pred_measurement @ self.covariance @ pred_measurement.T)
        meas_cov_inv = pred_meas_cov - pred_meas_cov @ (self.rotation.T @ self.depth_measurement_noise @ self.rotation + pred_meas_cov) @ pred_meas_cov

        # Kalman gain
        kalman_gain = self.covariance @ pred_measurement.T @ meas_cov_inv
        state_gain = kalman_gain[:9]
        bias_gain = kalman_gain[9:]

        self.state = expm(self._wedge(state_gain @ V)) @ self.state
        self.bias = self.bias + (bias_gain @ V)

        self.covariance = (np.eye(15) - kalman_gain @ pred_measurement) @ self.covariance
        return self.state, self.covariance

    def update_dvl(self, z: Vec3):
        dvl_rotation_body = np.eye(3)
        dvl_position_body = np.zeros((3, 1))
        measurement = dvl_rotation_body @ z.as_matrix() + dvl_position_body @ (self.last_controlInput['linear_acceleration'] - self.bias[0])

        zeros = np.zeros((3,3))
        measurement_jacobian = np.block([zeros, zeros, np.eye(3), zeros, zeros])

        pred_measurement = measurement_jacobian @ block_diag(self._adjoint(self.state), np.eye(6))

        full_measurement = np.array([measurement[0], measurement[1], measurement[2], -1, 0])

        # Make innovation vector
        V = (self.state @ full_measurement)[:3]
        
        meas_cov = inv(pred_measurement @ self.covariance @ pred_measurement.T + self.rotation @ self.dvl_measurement_noise @ self.rotation.T) 

        kalman_gain = self.covariance @ pred_measurement.T @ meas_cov
        state_gain = kalman_gain[:9]
        bias_gain = kalman_gain[9:]

        self.state[-1] = expm(self._wedge(state_gain @ V)) @ self.state
        self.bias[-1] = self.bias + (bias_gain @ V)

        self.covariance[-1] = (np.eye(15) - kalman_gain @ pred_measurement) @ self.covariance

        return self.state, self.covariance
    
    def update_ahrs(self, z):
        pass

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
    def _adjoint(self, state):
        rotation = state[:3, :3]
        velocity = state[:3, 3]
        position = state[:3, 4]

        adj = np.zeros((9, 9))

        adj[:3, :3] = rotation
        adj[3:6, 3:6] = rotation
        adj[6:9, 6:9] = rotation

        adj[3:6, :3] = self._skew(velocity) @ rotation
        adj[6:9, :3] = self._skew(position) @ rotation

        return adj


    def _wedge(self, x):
        so_3 = self._skew(x[0:3])
        vel = x[3:6].reshape(-1, 1)
        pos = x[6:9].reshape(-1,1)
        
        return np.block([[so_3, vel, pos],
                        [np.zeros((2,5))]])


    def _vee(self, x):
        return np.array([x[2, 1], x[0, 2], x[1, 0]])

    def run_filter(self, sensor_data):
        predicted_states = [self.state]
        
        timestamps = [sensor_data[0].timestamp]
        last_timestamp = sensor_data[0].timestamp


        # TODO: sensor_data is expected to be a list of SensorData
        # data.depth is a float
        # data.dvl is Vec3 transformed to matrix in the function
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


            predicted_states.append(self.state.copy())
            timestamps.append(data.time)

        return predicted_states, timestamps


    @property
    def rotation(self):
        return self.state[:3, :3]
    
    @property
    def velocity(self):
        return self.state[:3, 3]

    @property
    def position(self):
        return self.state[:3, 4]
    
    @property
    def inverse_state(self):
        rotation_transpose = self.rotation.T
        return np.block([
            [rotation_transpose, -rotation_transpose @ self.velocity, -rotation_transpose @ self.position],
            [np.zeros((1, 3)), 1, 0],
            [np.zeros((1, 3)), 0, 1],
        ])
