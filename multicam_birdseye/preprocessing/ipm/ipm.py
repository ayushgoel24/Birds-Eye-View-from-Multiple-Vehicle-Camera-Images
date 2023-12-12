import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

from .camera import Camera
from .camera_config_loader import CameraConfigLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_img_pair", metavar="CAM IMG", nargs='*')
    parser.add_argument("-wm", type=float, default=20)
    parser.add_argument("-hm", type=float, default=40)
    parser.add_argument("-r", type=float, default=20)
    parser.add_argument("--drone", type=str)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--cc", action="store_true")
    parser.add_argument("-v", action="store_true")
    return parser.parse_args()


def create_output_directory(output_path):
    """ Create an output directory if it doesn't exist. """
    output_dir = os.path.abspath(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def calculate_output_resolution(camera_loader, args):
    """
    Calculate the output resolution for the images.

    :param camera_loader: CameraConfigLoader object containing camera configurations.
    :param args: Command line arguments.
    :return: Tuple representing the output resolution (width, height).
    """
    if camera_loader.drone_config:
        # Adjust to match drone image resolution if specified
        drone_config = camera_loader.drone_config
        output_width = int(2 * drone_config["py"])
        output_height = int(2 * drone_config["px"])
        dx = output_width / drone_config["fx"] * drone_config["ZCam"]
        dy = output_height / drone_config["fy"] * drone_config["ZCam"]
        px_per_m = (output_height / dy, output_width / dx)
    else:
        # Standard resolution calculation
        px_per_m = (args.r, args.r)
        output_width = int(args.wm * px_per_m[0])
        output_height = int(args.hm * px_per_m[1])

    return output_width, output_height


def setup_mapping_to_world_coordinates(output_res, drone_config, args):
    """
    Setup the mapping from the image plane to world coordinates.

    :param output_res: Output resolution (width, height).
    :param drone_config: Configuration of the drone camera, if available.
    :param args: Command line arguments.
    :return: Transformation matrix M.
    """
    shift = (output_res[0] / 2.0, output_res[1] / 2.0)
    if drone_config:
        # Adjust mapping if a drone camera is used
        px_per_m = (output_res[0] / (output_res[0] / drone_config["fy"] * drone_config["ZCam"]),
                    output_res[1] / (output_res[1] / drone_config["fx"] * drone_config["ZCam"]))
        shift = (shift[0] + drone_config["YCam"] * px_per_m[0], 
                 shift[1] - drone_config["XCam"] * px_per_m[1])

    M = np.array([[1.0 / px_per_m[1], 0.0, -shift[1] / px_per_m[1]],
                  [0.0, -1.0 / px_per_m[0], shift[0] / px_per_m[0]],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    return M


def print_homographies(IPMs, camera_img_pairs):
    """ Print the homography matrices. """
    for idx, ipm in enumerate(IPMs):
        print(f"OpenCV homography for {camera_img_pairs[2*idx+1]}:")
        print(ipm.tolist())


def process_images(camera_loader, cams, IPMs, args, output_dir):
    """
    Process the images based on the provided cameras and IPMs.

    :param camera_loader: CameraConfigLoader object containing camera configurations.
    :param cams: List of Camera objects.
    :param IPMs: List of Inverse Perspective Mapping matrices.
    :param args: Command line arguments.
    :param output_dir: Directory to save the processed images.
    """
    for image_tuple in tqdm(camera_loader.image_paths):
        filename = os.path.basename(image_tuple[0])

        # Load images
        images = [cv2.imread(img_path) for img_path in image_tuple]

        # Warp input images using IPM
        warped_images = [cv2.warpPerspective(img, IPM, output_dir, flags=cv2.INTER_LINEAR if not args.cc else cv2.INTER_NEAREST)
                         for img, IPM in zip(images, IPMs)]

        # Stitch separate images to total bird's-eye-view
        birds_eye_view = np.zeros(warped_images[0].shape, dtype=np.uint8)
        for warped_img in warped_images:
            mask = np.any(warped_img != (0, 0, 0), axis=-1)
            birds_eye_view[mask] = warped_img[mask]

        # Display or export bird's-eye-view
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, filename), birds_eye_view)
        else:
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
            cv2.imshow(filename, birds_eye_view)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Main script execution
def main():
    args = parse_arguments()  # Assuming a function to parse arguments
    camera_loader = CameraConfigLoader(args.camera_img_pair, args.drone, args.batch)
    
    export = bool(args.output)
    output_dir = create_output_directory(args.output) if export else None

    cams = [Camera(config) for config in camera_loader.camera_configs]
    output_res = calculate_output_resolution(camera_loader, args)
    M = setup_mapping_to_world_coordinates(output_res, camera_loader.drone_config, args)

    IPMs = [cam.calculate_ipm(M) for cam in cams]
    if args.v:
        print_homographies(IPMs, args.camera_img_pair)
        return

    process_images(camera_loader, cams, IPMs, args)

if __name__ == "__main__":
    main()
