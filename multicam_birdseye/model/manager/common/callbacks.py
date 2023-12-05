import os
import tensorflow

class TrainerCallbacks:

    # TODO: get from Cache Manager
    # @staticmethod
    # def _get_tensorboard_log_dir():
    #     return os.path.join(
    #         "logs",
    #         datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     )
    
    @staticmethod
    def create_tensorboard_callback(tensorboard_dir=None):
        return tensorflow.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            update_freq="epoch",
            profile_batch=0
        )

    @staticmethod
    def create_model_checkpoint_callback(checkpoint_dir=None, num_training_batches=0, save_interval=0):
        return tensorflow.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"), 
            save_freq=num_training_batches * save_interval, 
            save_weights_only=True
        )
    
    @staticmethod
    def create_best_model_checkpoint_callback(checkpoint_dir=None):
        return tensorflow.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_weights.hdf5"), 
            save_best_only=True, 
            monitor="val_mean_io_u_with_one_hot_labels", 
            mode="max",
            save_weights_only=True
        )
    
    @staticmethod
    def create_early_stopping_callback(patience=10):
        return tensorflow.keras.callbacks.EarlyStopping(
            monitor="val_mean_io_u_with_one_hot_labels",
            mode="max",
            patience=patience,
            verbose=1
        )