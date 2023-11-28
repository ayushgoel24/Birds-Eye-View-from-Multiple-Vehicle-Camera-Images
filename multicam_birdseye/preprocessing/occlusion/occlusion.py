import argparse
import cv2
import multiprocessing
import os
import tqdm
import yaml

from .camera import Camera
from .image_operations import ImageOperations

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Determines the areas not visible from vehicle cameras and removes them from drone camera footage.")
    parser.add_argument("img", help="segmented drone image")
    parser.add_argument("drone", help="drone camera config file")
    parser.add_argument("cam", nargs="+", help="camera config file")
    parser.add_argument("--batch", help="process folders of images instead of single images", action="store_true")
    parser.add_argument("--output", help="output directory to write output images to")
    return parser.parse_args()

def load_image_paths(args):
    """
    Load image paths for processing.
    """
    if not args.batch:
        return [os.path.abspath(args.img)]
    else:
        path = os.path.abspath(args.img)
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f[0] != "."]
    
def parse_camera_configs(drone_config_path, camera_config_paths):
    """
    Parse camera configurations from files.
    """
    with open(os.path.abspath(drone_config_path)) as stream:
        drone_config = yaml.safe_load(stream)
    
    camera_configs = [yaml.safe_load(open(os.path.abspath(path))) for path in camera_config_paths]
    
    return drone_config, camera_configs

def create_cameras(camera_configs, base_link, px_per_meter):
    """
    Create camera objects from configurations.
    """
    return [Camera(config, base_link, px_per_meter) for config in camera_configs]

def calculate_pixels_per_meter(image, drone_config):
    """
    Calculate the number of pixels per meter for the image based on the drone camera configuration.

    :param image: The input image.
    :param drone_config: The configuration of the drone camera.
    :return: A tuple representing the number of pixels per meter in the x and y dimensions.
    """
    dx_meters = image.shape[1] / drone_config["fx"] * drone_config["ZCam"]
    dy_meters = image.shape[0] / drone_config["fy"] * drone_config["ZCam"]
    return (image.shape[1] / dx_meters, image.shape[0] / dy_meters)

def calculate_base_link(image, drone_config, pixels_per_meter):
    """
    Calculate the base link position based on the drone configuration and pixel density.

    :param image: The input image.
    :param drone_config: The configuration of the drone camera.
    :param pixels_per_meter: The number of pixels per meter in the x and y dimensions.
    :return: A tuple representing the base link position in the image.
    """
    base_link_x = int(image.shape[1] / 2.0 - drone_config["XCam"] * pixels_per_meter[0])
    base_link_y = int(image.shape[0] / 2.0 + drone_config["YCam"] * pixels_per_meter[1])
    return (base_link_x, base_link_y)

def process_images_in_batch(image_paths, cameras, base_link, args):
    """
    Process images in batch.
    """
    print("Warning: This might take a long time. Are you sure you need to regenerate the occluded labels?")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap(lambda path: ImageOperations.process_single_image(path, cameras, base_link, args), image_paths), total=len(image_paths)):
            pass

def main():
    args = parse_arguments()
    image_paths = load_image_paths(args)

    drone_config, camera_configs = parse_camera_configs(args.drone, args.cam)

    input_image = cv2.imread(image_paths[0])
    px_per_meter = calculate_pixels_per_meter(input_image, drone_config)
    base_link = calculate_base_link(input_image, drone_config, px_per_meter)

    cameras = create_cameras(camera_configs, base_link, px_per_meter)

    if args.output and not os.path.exists(os.path.abspath(args.output)):
        os.makedirs(os.path.abspath(args.output))

    if args.batch:
        process_images_in_batch(image_paths, cameras, base_link, args)
    else:
        ImageOperations.process_single_image(image_paths[0], cameras, base_link, args)

