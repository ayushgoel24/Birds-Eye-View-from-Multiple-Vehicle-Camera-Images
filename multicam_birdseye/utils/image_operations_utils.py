import tensorflow

class ImageOperationsUtil:

    @staticmethod
    def load_image_op(filename):
        img = tensorflow.io.read_file(filename)
        img = tensorflow.image.decode_png(img, channels=3)
        return img
    
    @staticmethod
    def resize_image_op(img, fromShape, toShape, cropToPreserveAspectRatio=True, interpolation=tensorflow.image.ResizeMethod.BICUBIC):
        if not cropToPreserveAspectRatio:
            img = tensorflow.image.resize(img, toShape, method=interpolation)
        else:
            # first crop to match target aspect ratio
            fx = toShape[1] / fromShape[1]
            fy = toShape[0] / fromShape[0]
            relevantAxis = 0 if fx < fy else 1
            if relevantAxis == 0:
                crop = fromShape[0] * toShape[1] / toShape[0]
                img = tensorflow.image.crop_to_bounding_box(img, 0, int((fromShape[1] - crop) / 2), fromShape[0], int(crop))
            else:
                crop = fromShape[1] * toShape[0] / toShape[1]
                img = tensorflow.image.crop_to_bounding_box(img, int((fromShape[0] - crop) / 2), 0, int(crop), fromShape[1])
            # then resize to target shape
            img = tensorflow.image.resize(img, toShape, method=interpolation)

        return img
    
    @staticmethod
    def one_hot_encode_image_op(image, palette):
        one_hot_map = []
        for class_colors in palette:
            class_map = tensorflow.zeros(image.shape[0:2], dtype=tensorflow.int32)
            for color in class_colors:
                # find instances of color and append layer to one-hot-map
                class_map = tensorflow.bitwise.bitwise_or(class_map, tensorflow.cast(tensorflow.reduce_all(tensorflow.equal(image, color), axis=-1), tensorflow.int32))
            one_hot_map.append(class_map)

        # finalize one-hot-map
        one_hot_map = tensorflow.stack(one_hot_map, axis=-1)
        one_hot_map = tensorflow.cast(one_hot_map, tensorflow.float32)

        return one_hot_map
