import cv2
import numpy as np
import os
import skimage.draw

import constants

class ImageOperations:

    @staticmethod
    def __apply_flood_fill(start_pixel, fill_color, input_image, output_image):
        """
        Applies the flood fill algorithm to an image.

        :param start_pixel: The starting pixel for flood fill (x, y).
        :param fill_color: The color to apply.
        :param input_image: The input image.
        :param output_image: The output image where changes are made.
        """
        mask = np.zeros((input_image.shape[0] + 2, input_image.shape[1] + 2), np.uint8)
        flood_fill_flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(image=input_image, mask=mask, seedPoint=start_pixel, newVal=(255, 255, 255), loDiff=(1, 1, 1), upDiff=(1, 1, 1), flags=flood_fill_flags)
        output_image[np.where(mask[1:-1, 1:-1] == 255)] = fill_color

    @staticmethod
    def __is_out_of_bounds(pixel, image):
        """
        Checks if a pixel is out of bounds of the given image.

        :param pixel: The pixel to check (x, y).
        :param image: The image.
        :return: Boolean indicating if the pixel is out of bounds.
        """
        return not (0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0])
    
    @staticmethod
    def __process_blocking_label(pixel, label, stop_ray, stop_transfer, input_image, output_image):
        """
        Processes a blocking label encountered by the ray.

        :param pixel: Current pixel being processed.
        :param label: The label of the blocking object.
        :param stop_ray: Flag indicating if the ray should be stopped.
        :param stop_transfer: Flag indicating if the transfer should be stopped.
        :param input_image: The input image.
        :param output_image: The output image.
        :return: Tuple of updated stop_ray and stop_transfer flags.
        """
        if label == "car":
            if stop_transfer:
                return stop_ray, stop_transfer
            stop_transfer = True
        else:
            stop_ray = True

        if not np.array_equal(output_image[pixel[1], pixel[0], :], constants.COLORS[label]):
            ImageOperations.__apply_flood_fill(pixel, constants.COLORS[label], input_image, output_image)
        
        return stop_ray, stop_transfer
    
    @staticmethod
    def __transfer_pixel(pixel, input_image, output_image):
        """
        Transfers a pixel from the input image to the output image.

        :param pixel: The pixel to transfer (x, y).
        :param input_image: The input image.
        :param output_image: The output image.
        """
        output_image[pixel[1], pixel[0], :] = input_image[pixel[1], pixel[0], :]

    @staticmethod
    def __cast_ray(start_point, end_point, input_image, output_image):
        """
        Casts a ray from one point to another, processing the pixels along the way.

        :param start_point: Starting point of the ray (x, y).
        :param end_point: Ending point of the ray (x, y).
        :param input_image: The input image.
        :param output_image: The output image where changes are made.
        """
        ray_pixels = list(zip(*skimage.draw.line(*start_point, *end_point)))
        stop_ray, stop_transfer = False, False

        for pixel in ray_pixels:
            if ImageOperations.__is_out_of_bounds(pixel, input_image):
                continue

            for label in constants.BLOCKING_LABELS:
                if np.array_equal(input_image[pixel[1], pixel[0], :], constants.COLORS[label]):
                    stop_ray, stop_transfer = ImageOperations.__process_blocking_label(pixel, label, stop_ray, stop_transfer, input_image, output_image)
                    if stop_ray or stop_transfer:
                        break

            if stop_ray: break

            if not stop_transfer:
                ImageOperations.__transfer_pixel(pixel, input_image, output_image)

    @staticmethod
    def __is_ego_vehicle_in_image(position):
        """
        Checks if the ego vehicle is in the image.

        :param position: Position of the ego vehicle.
        :return: Boolean indicating if the ego vehicle is in the image.
        """
        return position[0] > 0 and position[1] > 0

    @staticmethod
    def __calculate_camera_rays(image, camera):
        """
        Calculates rays from the camera's perspective to the edges of the image.

        :param image: The input image.
        :param camera: The camera object.
        :return: List of rays (start and end points).
        """
        rays = []
        for x in range(image.shape[1]):
            if camera.can_see(x, 0):
                rays.append((camera.origin, (x, 0)))
            if camera.can_see(x, image.shape[0]):
                rays.append((camera.origin, (x, image.shape[0])))
        for y in range(image.shape[0]):
            if camera.can_see(0, y):
                rays.append((camera.origin, (0, y)))
            if camera.can_see(image.shape[1], y):
                rays.append((camera.origin, (image.shape[1], y)))
        return rays
    
    @staticmethod
    def __collect_visibility_rays(image, camera_list):
        """
        Collects rays for visibility determination from each camera's perspective.

        :param image: The input image.
        :param camera_list: List of cameras.
        :return: List of rays (start and end points).
        """
        rays = []
        for camera in camera_list:
            rays.extend(ImageOperations.__calculate_camera_rays(image, camera))
        return rays

    @staticmethod
    def __export_or_display_output_image(output_image, filename):
        """
        Exports or displays the output image.

        :param output_image: The processed output image.
        :param filename: Filename for saving or displaying the image.
        """
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if args.output:
            cv2.imwrite(os.path.join(output_dir, filename), output_image)
        else:
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
            cv2.imshow(filename, output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    @staticmethod
    def process_single_image(image_path, cameras, base_link, args):
        """
        Processes a single image by casting rays to determine visibility from cameras and applying transformations.

        :param image_path: Path to the image to be processed.
        :param cameras: List of camera objects.
        :param base_link: Base link position of the drone in the image.
        :param args: Command-line arguments.
        """
        filename = os.path.basename(image_path)

        # Read input image and create a blank output image
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        output_image = np.full(input_image.shape, constants.COLORS["occluded"], dtype=np.uint8)

        # Recolor ego vehicle temporarily, so it does not block visibility
        if ImageOperations.__is_ego_vehicle_in_image(base_link):
            ImageOperations.__apply_flood_fill(base_link, constants.DUMMY_COLOR, input_image, input_image)

        # Collect and cast rays for visibility determination
        rays = ImageOperations.__collect_visibility_rays(input_image, cameras)
        for ray_start, ray_end in rays:
            ImageOperations.__cast_ray(ray_start, ray_end, input_image, output_image)

        # Recolor ego vehicle back to original color and transfer to output
        if ImageOperations.__is_ego_vehicle_in_image(base_link):
            ImageOperations.__apply_flood_fill(base_link, constants.COLORS["car"], input_image, output_image)

        # Export or display the processed image
        ImageOperations.__export_or_display_output_image(output_image, filename, args.output)