import torch
import torch.utils.data as data
import numpy as np
import os
import re  # For regex to extract labels

class UCFDataSet(data.Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.files = os.listdir(datapath)

    def __len__(self):
        return len(self.files)

    def process_feat(self, feat, length):
        new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
        r = np.linspace(0, len(feat), length + 1, dtype=np.int64)

        for i in range(length):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], axis=0)
            else:
                new_feat[i, :] = feat[r[i], :]

        return new_feat

    def extract_label(self, filename):
        # Use regex to find all characters up to the first digit
        match = re.match(r"([^\d]*)", filename)  # Matches all characters until a digit
        return match.group(0) if match else ""

    def __getitem__(self, idx):
        file_path = os.path.join(self.datapath, self.files[idx])
        features = np.load(file_path)  # Load each file on demand
        features = features.transpose(1, 0, 2)
        
        divided_feature = []
        for feature in features:
            feature = self.process_feat(feature, 32)  # Process each feature
            divided_feature.append(feature)

        divided_feature = np.array(divided_feature, dtype=np.float32)

        # Reshape to (32, 2048) for each sample
        features_tensor = torch.tensor(divided_feature).reshape(-1, 2048)  # Converts directly to tensor

        # Extract label from the filename
        label = self.extract_label(self.files[idx])
        
        return features_tensor, label  # Return both features and labels

def main():
    datapath = '/data/ucfdata/UCF_Train_ten_crop_i3d'
    dataset = UCFDataSet(datapath)

    BATCH_SIZE = 16  # Define your batch size
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Example usage
    for batch_features, batch_labels in dataloader:
        print(batch_features.shape)  # Should print (BATCH_SIZE * 32, 2048)
        print(batch_labels)  # Prints the labels

if __name__ == '__main__':
    main()
