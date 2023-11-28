import cv2
import numpy as np
import skimage.draw

import constants

class ImageOperations:

    @staticmethod
    def apply_flood_fill(start_pixel, fill_color, input_image, output_image):
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
    def is_out_of_bounds(pixel, image):
        """
        Checks if a pixel is out of bounds of the given image.

        :param pixel: The pixel to check (x, y).
        :param image: The image.
        :return: Boolean indicating if the pixel is out of bounds.
        """
        return not (0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0])
    
    @staticmethod
    def process_blocking_label(pixel, label, stop_ray, stop_transfer, input_image, output_image):
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
            ImageOperations.apply_flood_fill(pixel, constants.COLORS[label], input_image, output_image)
        
        return stop_ray, stop_transfer
    
    @staticmethod
    def transfer_pixel(pixel, input_image, output_image):
        """
        Transfers a pixel from the input image to the output image.

        :param pixel: The pixel to transfer (x, y).
        :param input_image: The input image.
        :param output_image: The output image.
        """
        output_image[pixel[1], pixel[0], :] = input_image[pixel[1], pixel[0], :]

    @staticmethod
    def cast_ray(start_point, end_point, input_image, output_image):
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
            if ImageOperations.is_out_of_bounds(pixel, input_image):
                continue

            for label in constants.BLOCKING_LABELS:
                if np.array_equal(input_image[pixel[1], pixel[0], :], constants.COLORS[label]):
                    stop_ray, stop_transfer = ImageOperations.process_blocking_label(pixel, label, stop_ray, stop_transfer, input_image, output_image)
                    if stop_ray or stop_transfer:
                        break

            if stop_ray: break

            if not stop_transfer:
                ImageOperations.transfer_pixel(pixel, input_image, output_image)