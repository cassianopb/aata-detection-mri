import numpy as np
import torch
import torch.nn as nn
import timm


class KneeClassifier(nn.Module):
    """
    Binary classifier for knee MRI slice-level AATA detection.

    Uses a ResNet10T backbone pretrained on ImageNet, with a lightweight
    custom head for binary classification (AATA present / absent).
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        base = timm.create_model("resnet10t", pretrained=True)
        self.base_model = base
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def generate_gradcam(
    model: KneeClassifier,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """
    Compute a Grad-CAM saliency map for a single input tensor.

    Args:
        model: KneeClassifier in eval mode.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_layer: Convolutional layer from which activations are extracted
                      (typically model.base_model.layer4).

    Returns:
        Normalized Grad-CAM heatmap as a 2-D float32 array in [0, 1].
    """
    activations, gradients = [], []

    fwd_handle = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o)
    )
    bwd_handle = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0])
    )

    output = model(input_tensor)
    model.zero_grad()
    output[:, 0].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    acts = activations[0].squeeze().cpu().detach().numpy()
    grads = gradients[0].squeeze().cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for w, a in zip(weights, acts):
        cam += w * a

    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8
    return cam
