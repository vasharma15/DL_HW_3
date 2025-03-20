from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Third convolutional layer

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming the input images are 32x32
        self.fc2 = nn.Linear(512, num_classes)  # Output layer for classification

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        z = self.pool(F.relu(self.conv1(z)))  # First conv layer
        z = self.pool(F.relu(self.conv2(z)))  # Second conv layer
        z = self.pool(F.relu(self.conv3(z)))  # Third conv layer

        z = z.view(-1, 128 * 8 * 8)
        z = F.relu(self.fc1(z))

        # TODO: replace with actual forward pass
        logits = self.fc2(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Additional conv layer
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Additional conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Additional bottleneck processing for better feature extraction
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path) with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # After skip connection (32+32=64 channels)
        self.conv_after_up1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # After second skip connection (32+3=35 channels)
        self.conv_after_up2 = nn.Sequential(
            nn.Conv2d(35, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Output heads
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        input_x = x  # Save for skip connection

        # Encoder
        d1 = self.down1(x)  # (B, 32, 48, 64)
        d2 = self.down2(d1)  # (B, 64, 24, 32)

        # Bottleneck
        b = self.bottleneck(d2)  # (B, 64, 24, 32)

        # Decoder with skip connections
        u1 = self.up1(b)  # (B, 32, 48, 64)
        # Skip connection from down1 to up1
        u1_combined = torch.cat([u1, d1], dim=1)  # (B, 64, 48, 64)
        u1_processed = self.conv_after_up1(u1_combined)  # (B, 32, 48, 64)

        u2 = self.up2(u1_processed)  # (B, 32, 96, 128)
        # Skip connection from input to up2
        u2_combined = torch.cat([u2, input_x], dim=1)  # (B, 35, 96, 128)
        u2_processed = self.conv_after_up2(u2_combined)  # (B, 16, 96, 128)

        # Task-specific outputs
        logits = self.seg_head(u2_processed)  # (B, 3, 96, 128)
        depth = self.depth_head(u2_processed)  # (B, 1, 96, 128)

        return logits, depth.squeeze(1)


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Used for inference, takes an image and returns class labels and normalized depth.
            This is what the metrics use as input (this is what the grader will use!).

            Args:
                x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

            Returns:
                tuple of (torch.LongTensor, torch.FloatTensor):
                    - pred: class labels {0, 1, 2} with shape (b, h, w)
                    - depth: normalized depth [0, 1] with shape (b, h, w)
            """
            logits, raw_depth = self(x)
            pred = logits.argmax(dim=1)

            # Optional additional post-processing for depth only if needed
            depth = raw_depth

            return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        print(model_path)
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
