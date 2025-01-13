from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # Hugging Face Dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  # Access a single data item as a dictionary
        image, label = item['image'], item['label']

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
