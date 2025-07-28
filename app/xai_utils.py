import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

def gradcam_explain(model, input_tensor, target_layer, target_class):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    image_rgb = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())

    cam_image = show_cam_on_image(image_rgb, grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_image)

def lime_explain(model, pil_image, transform, target_class, device):
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        imgs = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return probs

    explanation = explainer.explain_instance(
        np.array(pil_image),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    return Image.fromarray(lime_img)

def occlusion_sensitivity_analysis(model, pil_image, transform, target_class, device):
    model.eval()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Vorhersage auf Original
    with torch.no_grad():
        base_prob = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()

    # Occlusion Map
    mask_size = 20
    stride = 20
    width, height = 256, 256
    heatmap = np.zeros((height // stride, width // stride))

    for y in range(0, height - mask_size, stride):
        for x in range(0, width - mask_size, stride):
            occluded = input_tensor.clone()
            occluded[:, :, y:y+mask_size, x:x+mask_size] = 0
            with torch.no_grad():
                prob = torch.softmax(model(occluded), dim=1)[0, target_class].item()
            heatmap[y // stride, x // stride] = base_prob - prob

    # Heatmap als Bild
    from matplotlib import pyplot as plt
    import io

    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap="jet", interpolation="bilinear")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf)