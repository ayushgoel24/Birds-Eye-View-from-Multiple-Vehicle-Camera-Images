import tensorflow

class MetricUtil:

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        def wcce(y_true, y_pred):
            Kweights = tensorflow.constant(weights)
            if not tensorflow.is_tensor(y_pred): y_pred = tensorflow.constant(y_pred)
            y_true = tensorflow.cast(y_true, y_pred.dtype)
            return tensorflow.keras.backend.categorical_crossentropy(y_true, y_pred) * tensorflow.keras.backend.sum(y_true * Kweights, axis=-1)
        return wcce