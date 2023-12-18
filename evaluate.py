import cache
import multicam_birdseye


if __name__ == '__main__':
    datasetConfigurations = cache.DatasetConfigurationManager(config_file="configurations/dataset_configurations.yaml")
    datasetConfigurations.load_config()

    multicam_birdseye.model.manager.Evaluator(
        validation_input_paths=datasetConfigurations.config["validation"]["data-path"],
        validation_label_path=datasetConfigurations.config["validation"]["label-path"],
        max_validation_samples=datasetConfigurations.config["validation"]["max-samples"],
        target_image_shape=datasetConfigurations.config["image"]["shape"],
        input_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["data"],
        label_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["label"],
        class_names=datasetConfigurations.config["validation"]["class-names"],
        model_path=datasetConfigurations.config["model"]["path"],
        homography_path=datasetConfigurations.config["homography"]["path"],
        pretrained_model_weights=datasetConfigurations.config["model"]["pretrained-weights"]
    ).run()