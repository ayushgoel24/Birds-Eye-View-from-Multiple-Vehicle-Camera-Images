import os
import yaml

class CameraConfigLoader:
    """ 
    This class handles the loading of camera configurations and image paths.
    """

    def __init__(self, camera_img_pairs, drone_config_path=None, batch_mode=False):
        self.camera_img_pairs = camera_img_pairs
        self.drone_config_path = drone_config_path
        self.batch_mode = batch_mode
        self.camera_configs = []
        self.image_paths = []
        self.load_configs_and_paths()

    def load_configs_and_paths(self):
        """ Load camera configurations and image paths from the provided arguments. """
        for idx in range(int(len(self.camera_img_pairs) / 2)):
            config_path = os.path.abspath(self.camera_img_pairs[2 * idx])
            image_path = self.camera_img_pairs[2 * idx + 1]
            with open(config_path) as stream:
                self.camera_configs.append(yaml.safe_load(stream))
            if not self.batch_mode:
                self.image_paths.append([image_path])
            else:
                self.image_paths.append(self._load_batch_images(image_path))

        if self.drone_config_path:
            with open(os.path.abspath(self.drone_config_path)) as stream:
                self.drone_config = yaml.safe_load(stream)
        else:
            self.drone_config = None

    def _load_batch_images(self, path):
        """ Load batch images from the specified directory. """
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f[0] != '.']

    @property
    def output_filenames(self):
        """ Get output filenames based on the image paths. """
        return [os.path.basename(image_tuple[0]) for image_tuple in self.image_paths]