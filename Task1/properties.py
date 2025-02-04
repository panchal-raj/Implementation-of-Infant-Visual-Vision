from imports import *


# Restructure the dataset
class VCASDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes (subdirectories).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List to store image paths and corresponding labels

        # Load all image paths and their respective labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Ensure images are Grayscale
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}


class Utility:
    def __init__(self, age, image_path):
        self.age = age
        self.image_path = image_path

    def get_sigma(self):
        '''
        Parameters:
        None

        Output:
        returns the estimate value of sigma corresponding to the blur in visual acuity.
        '''
        # Validate that age is non-negative
        if self.age < 0:
            raise ValueError("Age cannot be negative. Please provide a valid age in months.")

        if self.age <= 1:
            # Snellen - 20/600 (newborn to 1 month)
            snellen, sigma = "20/600", 4
        elif 1 < self.age <= 4:
            # Snellen - 20/480 (1 to 4 months)
            snellen, sigma = "20/489", 3
        elif 4 < self.age <= 8:
            # Snellen - 20/373 (4 to 8 months)
            snellen, sigma = "20/373", 2
        elif 8 < self.age <= 18:
            # Snellen - 20/221 (8 to 18 months)
            snellen, sigma = "20/221", 1
        else:
            # Snellen - 20/20 (18+ months)
            snellen, sigma = "20/20", 0

        return snellen, sigma

    def get_kernel_size(self, sigma):
        '''
        Parameters:
        sigma: Standard deviation for the gaussian blur kernel.

        Output:
        Returns the nearest odd kernel size corresponding to the value of sigma.
        '''
        kernel_size = max(1, int(6 * sigma))  # Ensure a minimum kernel size of 1
        if kernel_size % 2 == 0:  # Ensure the kernel size is odd
            kernel_size += 1
        return kernel_size

    def get_cpd(self):
        """
        Output:
        cutoff_frequency: The frequency threshold for the given age.

        Map age to cutoff frequency for CSF (low-pass filter).
        Contrast sensitivity increases with age.
        """
        if self.age < 0:
            raise ValueError("Age cannot be negative.")

        if 0 < self.age <= 1:
            cpd = 2.4
        elif 1 < self.age <= 2:
            cpd = 2.8
        elif 2 < self.age <= 3:
            cpd = 4.0
        elif 3 < self.age <= 6:
            cpd = 8.0
        elif 6 < self.age <= 12:
            cpd = 10.0
        elif 12 < self.age <= 72:
            cpd = 20.0
        elif 72 < self.age <= 240:
            cpd = 32.0
        else:
            cpd = 30.0

        return cpd

    def load_image(self):
        """
        Given the path of the image, loads it if the input is a path string
        else if it's a tensor simply returns it.
        """
        # If it's a file path, open the image
        if isinstance(self.image_path, str):
            original_image = Image.open(self.image_path).convert('L')
        elif isinstance(self.image_path, torch.Tensor):
            # If it's already a tensor and convert tensor to a PIL Image
            original_image = transforms.ToPILImage()(self.image_path.squeeze(0))
        else:
            # In this case this is a image matrix
            return self.image_path

        return original_image


class VisualAcuity(object):
    def __init__(self, age):
        self.age = age

    def __call__(self, image_path):
        '''
        Parameters:
        None
        Output:
        maps age to the snellen chart value and then maps this
        snellen chart value to gaussian kernel and apply the
        similar blur on the images and produces the output.
        '''
        age_params = Utility(age=self.age, image_path=image_path)

        # Load the original image
        original_image = age_params.load_image()

        # Convert the image to a PyTorch tensor and add batch and channel dimensions
        original_tensor = transforms.ToTensor()(original_image).unsqueeze(0)

        # Apply Gaussian blur
        _, sigma = age_params.get_sigma()
        kernel_size = age_params.get_kernel_size(sigma)

        # Apply Gaussian blur to the image tensor
        if sigma > 0:
            blurred_image = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(original_tensor)
        else:
            # If sigma is 0, the blurred image is the same as the original
            blurred_image = original_tensor

        return blurred_image


class ContrastSensitivity(object):
    def __init__(self, age):
        self.age = age

    def __call__(self, image_path):
        """
        Parameters:
        image: Grayscale image as a numpy array.
        age: (age in months) to set CSF cutoff frequency.
        Output:
        filtered image after applying the CSF-based low-pass filter.
        """
        age_params = Utility(age=self.age, image_path=image_path)

        # Load the original image
        original_image = age_params.load_image()

        image = np.array(original_image)

        # Get cutoff frequency based on age group
        cutoff_frequency = age_params.get_cpd()

        # Fourier Transform to move to frequency domain
        f_transform = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to center

        # Create a low-pass filter mask
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), dtype=np.uint8)

        # Define the mask as a circular low-pass filter based on cutoff frequency
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if distance <= cutoff_frequency:
                    mask[i, j] = 1

        # Apply mask to the frequency domain image
        f_shifted = f_shifted * mask

        # Inverse FFT to return to spatial domain
        f_ishift = np.fft.ifftshift(f_shifted)
        filtered_image = np.fft.ifft2(f_ishift)
        filtered_image = np.abs(filtered_image)

        return torch.from_numpy(filtered_image)


class DefineDataLoader:
    def __init__(self, age, root_dir=ROOT_DIR):
        self.age = age
        self.root_dir = root_dir

    def construct_dataloader(self):
        # Transformation for visual acuity
        transform_va = transforms.Compose([
            transforms.ToTensor(),
            VisualAcuity(age=self.age),
        ])

        # Transformation for contrast sensitivity
        transform_cs = transforms.Compose([
            transforms.ToTensor(),
            ContrastSensitivity(age=self.age),
        ])

        # Combined transformations
        transform_combined = transforms.Compose([
            transforms.ToTensor(),
            VisualAcuity(age=self.age),
            ContrastSensitivity(age=self.age),
        ])

        # Without transformations
        transform_original = transforms.Compose([
            transforms.ToTensor()
        ])

        # Initialize the datasets
        dataset_va = VCASDataset(root_dir=self.root_dir, transform=transform_va)  # With visual acuity
        dataset_cs = VCASDataset(root_dir=self.root_dir, transform=transform_cs)  # With contrast sensitivity
        dataset_combined = VCASDataset(root_dir=self.root_dir,
                                       transform=transform_combined)  # With both transformations
        dataset_original = VCASDataset(root_dir=self.root_dir, transform=transform_original)  # Without transformations

        # Create the DataLoader
        dataloader_va = DataLoader(dataset_va, batch_size=4, shuffle=True, num_workers=2)
        dataloader_cs = DataLoader(dataset_cs, batch_size=4, shuffle=True, num_workers=2)
        dataloader_combined = DataLoader(dataset_combined, batch_size=4, shuffle=True, num_workers=2)
        dataloader_original = DataLoader(dataset_original, batch_size=4, shuffle=True, num_workers=2)

        return [dataloader_va, dataloader_cs, dataloader_combined, dataloader_original]

    def dataloader_runtime(self, dataloader_original, dataloader_prop, property):
        # Measure time for the DataLoader without transforms
        print("Measuring time for DataLoader without transforms:")
        start_time_original = time.time()
        for i, sample in enumerate(tqdm(dataloader_original, desc="Loading original data")):
            images, labels = sample['image'], sample['label']
        end_time_original = time.time()
        time_original = end_time_original - start_time_original
        print(f"Time taken for DataLoader without transforms: {time_original:.2f} seconds")

        # Measure time for the DataLoader with transformation
        print(f"\nMeasuring time for DataLoader with {property} transformation:")
        start_time_transformed = time.time()
        for i, sample in enumerate(tqdm(dataloader_prop, desc=f"Loading transformed data with {property}")):
            images, labels = sample['image'], sample['label']
        end_time_transformed = time.time()
        time_transformed = end_time_transformed - start_time_transformed
        print(f"\nTime taken for DataLoader with {property} transformation: {time_transformed:.2f} seconds")

        # Compare the results with and without property
        print(f"Comparison of time taken for {property}:")
        print(f"With transform: {time_transformed:.2f} seconds")
        print(f"Without transform: {time_original:.2f} seconds")
        print(f"Difference: {abs(time_transformed - time_original):.2f} seconds\n")

        return time_original, time_transformed


    def visualize_batch(self, dataloader, property):
        # Function to visualize a batch of images

        # Get a batch of data
        batch = next(iter(dataloader))

        # Extract images and labels from the batch
        images, labels = batch['image'], batch['label']

        # Check the size of the image batch
        print(f"Batch shape for {property}: {images.shape}")
        print(f"Labels for {property}: {labels}")

        # Convert images from tensor to numpy for visualization
        images = images.numpy()  # Convert from torch tensor to numpy array
        images = images.squeeze()  # Remove channel dimension if grayscale (will make shape [batch_size, height, width])

        # Plot the images
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        plt.suptitle(f"{property}")
        for i, ax in enumerate(axes):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')

        plt.show()


    def visualize_runtime(self, time_original, time_va, time_cs, time_combined):
        # Categories and times
        categories = ['Original', 'Visual Acuity', 'Contrast Sensitivity', 'Both Transforms']
        times = [time_original, time_va, time_cs, time_combined]

        # Color scheme (Colorblind-friendly palette)
        colors = ['#0072B2', '#56B4E9', '#E69F00', '#F0E442']

        # Figure dimensions and background
        plt.figure(figsize=(10, 7), dpi=100)
        plt.gca().set_facecolor('#F9F9F9')  # Light gray background for the plot
        plt.grid(axis='y', linestyle='--', alpha=0.6, color='gray')  # Add gridlines for clarity

        # Create bars
        bars = plt.bar(categories, times, color=colors, edgecolor='black', linewidth=1.2)

        # Add shadow-like 3D effect to bars
        for bar in bars:
            bar.set_alpha(0.9)  # Adjust bar transparency
            plt.gca().bar(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.1, width=0.5, alpha=0.2,
                          color=bar.get_facecolor())

        # Add value annotations on top of bars
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X position
                bar.get_height() + 0.1,  # Y position
                f'{bar.get_height():.2f}s',  # Text to display
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333'
            )

        # Configure the axes
        plt.xticks(fontsize=12, fontweight='bold', color='#333')  # Format X-axis labels
        plt.yticks(fontsize=12, fontweight='bold', color='#333')  # Format Y-axis labels
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # Limit Y-axis to integer values

        # Add title and axis labels
        plt.title('Dataloader Performance Comparison', fontsize=18, fontweight='bold', color='#333', pad=20)
        plt.xlabel('Transformation Type', fontsize=14, fontweight='bold', color='#333', labelpad=15)
        plt.ylabel('Total Time (seconds)', fontsize=14, fontweight='bold', color='#333', labelpad=15)

        # Add legend
        plt.legend(
            bars,
            categories,
            title='Transformations',
            loc='upper left',
            fontsize=12,
            title_fontsize=14,
            frameon=True,
            edgecolor='black'
        )

        # Add a footer (explanatory text at the bottom of the plot)
        plt.figtext(
            0.5, -0.05,
            'Data represents total loading times for dataloaders with different transformations.',
            ha='center', fontsize=10, color='gray', fontstyle='italic'
        )

        # Remove top and right frame spines for a modern look
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)

        # Final adjustments
        plt.tight_layout()
        plt.show()

