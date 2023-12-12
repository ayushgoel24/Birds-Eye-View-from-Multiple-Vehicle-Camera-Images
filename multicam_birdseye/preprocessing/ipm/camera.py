import numpy as np

class Camera:
    """
    This class represents a camera with intrinsic and extrinsic parameters.
    It provides methods to set camera matrix (K), rotation matrix (R), translation vector (t),
    and update the projection matrix (P).
    """

    def __init__(self, config):
        """
        Initializes the Camera object with given configuration.
        :param config: Dictionary containing camera configuration parameters.
        """
        self.K = np.zeros([3, 3])
        self.R = np.zeros([3, 3])
        self.t = np.zeros([3, 1])
        self.P = np.zeros([3, 4])
        self.set_intrinsics(config["fx"], config["fy"], config["px"], config["py"])
        self.set_extrinsics(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]), config["XCam"], config["YCam"], config["ZCam"])
        self.update_projection_matrix()

    def set_intrinsics(self, fx, fy, px, py):
        """
        Sets the intrinsic camera matrix K.
        :param fx: Focal length in x-axis.
        :param fy: Focal length in y-axis.
        :param px: Principal point x-coordinate.
        :param py: Principal point y-coordinate.
        """
        self.K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

    def set_extrinsics(self, yaw, pitch, roll, XCam, YCam, ZCam):
        """
        Sets the extrinsic parameters of the camera (rotation and translation).
        :param yaw: Yaw angle in radians.
        :param pitch: Pitch angle in radians.
        :param roll: Roll angle in radians.
        :param XCam: X coordinate of the camera position.
        :param YCam: Y coordinate of the camera position.
        :param ZCam: Z coordinate of the camera position.
        """
        Rz = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(-pitch), 0, np.sin(-pitch)], [0, 1, 0], [-np.sin(-pitch), 0, np.cos(-pitch)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(-roll), -np.sin(-roll)], [0, np.sin(-roll), np.cos(-roll)]])
        Rs = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # Axis switch
        self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))
        X = np.array([XCam, YCam, ZCam])
        self.t = -self.R.dot(X)

    def update_projection_matrix(self):
        """
        Updates the projection matrix P of the camera.
        """
        Rt = np.hstack((self.R, self.t.reshape(3, 1)))
        self.P = self.K.dot(Rt)