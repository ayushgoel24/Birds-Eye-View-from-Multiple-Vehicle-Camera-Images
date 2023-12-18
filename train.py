import cache
import multicam_birdseye


if __name__ == '__main__':
    datasetConfigurations = cache.DatasetConfigurationManager(config_file="configurations/dataset_configurations.yaml")
    datasetConfigurations.load_config()

    multicam_birdseye.model.manager.Trainer(
        training_input_paths=datasetConfigurations.config["training"]["data-path"],
        training_label_path=datasetConfigurations.config["training"]["label-path"],
        max_training_samples=datasetConfigurations.config["training"]["max-samples"],
        validation_input_paths=datasetConfigurations.config["validation"]["data-path"],
        validation_label_path=datasetConfigurations.config["validation"]["label-path"],
        max_validation_samples=datasetConfigurations.config["validation"]["max-samples"],
        target_image_shape=datasetConfigurations.config["image"]["shape"],
        input_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["data"],
        label_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["label"],
        model_path=datasetConfigurations.config["model"]["path"],
        homography_path=datasetConfigurations.config["homography"]["path"],
        training_epochs=datasetConfigurations.config["training"]["epochs"],
        batch_size=datasetConfigurations.config["training"]["batch-size"],
        learning_rate=datasetConfigurations.config["training"]["learning-rate"],
        class_loss_weights=datasetConfigurations.config["training"]["class-loss-weights"],
        early_stopping_patience=datasetConfigurations.config["training"]["early-stopping-patience"],
        model_save_interval=datasetConfigurations.config["training"]["model-save-interval"],
        output_directory=datasetConfigurations.config["training"]["output-directory"],
        pretrained_model_weights=datasetConfigurations.config["model"]["pretrained-weights"]
    ).run()