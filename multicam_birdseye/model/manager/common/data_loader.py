import numpy as np

from ....utils import PathUtil

class DataLoader:

    @staticmethod
    def prepare_data(data_paths, label_path, max_samples):
        input_files = [PathUtil.get_files_in_folder(folder) for folder in data_paths]
        label_files = PathUtil.get_files_in_folder(label_path)

        _, indices = PathUtil.sample_list(label_files, n_samples=max_samples)
        sampled_inputs = [np.take(files, indices) for files in input_files]
        sampled_labels = np.take(label_files, indices)
        
        input_shape = PathUtil.load_image(sampled_inputs[0][0]).shape[0:2]
        label_shape = PathUtil.load_image(sampled_labels[0]).shape[0:2]

        print(f"Found {len(sampled_labels)} samples")
        return sampled_inputs, sampled_labels, input_shape, label_shape