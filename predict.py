import cache
import multicam_birdseye


if __name__ == '__main__':
    datasetConfigurations = cache.DatasetConfigurationManager(config_file="configurations/dataset_configurations.yaml")
    datasetConfigurations.load_config()

    multicam_birdseye.model.manager.Predictor(
        test_input_paths=datasetConfigurations.config["test"]["data-path"],
        max_test_samples=datasetConfigurations.config["test"]["max-samples"],
        target_image_shape=datasetConfigurations.config["image"]["shape"],
        input_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["data"],
        label_one_hot_palette_path=datasetConfigurations.config["one-hot-palette"]["label"],
        class_names=datasetConfigurations.config["test"]["class-names"],
        model_path=datasetConfigurations.config["model"]["path"],
        homography_path=datasetConfigurations.config["homography"]["path"],
        pretrained_model_weights=datasetConfigurations.config["model"]["pretrained-weights"]
    ).run()