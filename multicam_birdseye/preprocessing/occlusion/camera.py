import numpy as np

class Camera:
    """Camera class representing a vehicle camera."""

    def __init__(self, config, frame, pixels_per_meter):
        """
        Initialize the camera with configuration, frame reference, and pixel density.

        :param config: Camera configuration.
        :param frame: Frame reference.
        :param pixels_per_meter: Pixels per meter for image resolution.
        """
        self.origin = (frame[0] + config["XCam"] * pixels_per_meter[0], frame[1] - config["YCam"] * pixels_per_meter[1])
        self.yaw = -config["yaw"]
        self.fov = 2.0 * np.arctan(config["px"] / config["fx"]) * 180.0 / np.pi
        self.fov_bounds = self.calculate_fov_bounds()

    def calculate_fov_bounds(self):
        """Calculate the field of view bounds of the camera."""
        theta_min = (self.yaw - self.fov / 2.0) % 180 if self.yaw - self.fov / 2.0 < -180 else self.yaw - self.fov / 2.0
        theta_max = (self.yaw + self.fov / 2.0) % -180 if self.yaw + self.fov / 2.0 > 180 else self.yaw + self.fov / 2.0
        return theta_min, theta_max

    def can_see(self, x, y):
        """
        Determine if the camera can see the point (x, y).

        :param x: X-coordinate of the point.
        :param y: Y-coordinate of the point.
        :return: Boolean indicating if the point is within the camera's FOV.
        """
        dx, dy = x - self.origin[0], y - self.origin[1]
        theta = np.arctan2(dy, dx) * 180.0 / np.pi
        if self.fov_bounds[0] > self.fov_bounds[1]:
            return (self.fov_bounds[0] <= theta) or (theta <= self.fov_bounds[1])
        else:
            return (self.fov_bounds[0] <= theta) and (theta <= self.fov_bounds[1])
