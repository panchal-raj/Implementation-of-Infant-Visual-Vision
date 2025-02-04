from imports import *
from transforms import ContrastAdjust,AcuityBlur

def withoutTransforms():
    # Create DataLoaders for each age group
    train_loaders, val_loaders = [], []

    for i, age_in_months in enumerate(age_groups):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_data, val_data = random_split(base_dataset, [train_size, val_size])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Create DataLoaders for the current subset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

def BothTransformsLoader():
    # Create DataLoaders for each age group
    train_loaders, val_loaders = [], []

    for i, age_in_months in enumerate(age_groups):
        transform = transforms.Compose([
            AcuityBlur(age_in_months=age_in_months),
            ContrastAdjust(age_in_months=age_in_months), 
            transforms.ToTensor()
        ])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_data, val_data = random_split(base_dataset, [train_size, val_size])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Create DataLoaders for the current subset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

def AcuityTransformLoader():
    # Create DataLoaders for each age group
    train_loaders, val_loaders = [], []

    for i, age_in_months in enumerate(age_groups):
        transform = transforms.Compose([
            AcuityBlur(age_in_months=age_in_months),
            transforms.ToTensor()
        ])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_data, val_data = random_split(base_dataset, [train_size, val_size])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Create DataLoaders for the current subset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        
    return train_loaders, val_loaders

def ContrastTransformLoader():
    # Create DataLoaders for each age group
    train_loaders, val_loaders = [], []

    for i, age_in_months in enumerate(age_groups):
        transform = transforms.Compose([
            ContrastAdjust(age_in_months=age_in_months), 
            transforms.ToTensor()
        ])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_data, val_data = random_split(base_dataset, [train_size, val_size])
        # Apply the transformation to the base dataset
        base_dataset.transform = transform

        # Create DataLoaders for the current subset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        
    return train_loaders, val_loaders
