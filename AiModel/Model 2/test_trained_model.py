import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os
import numpy as np
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# -------- CONFIG --------
IMG_SIZE = 299
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r'D:/Kaif/Hackathon25/CDAC/AiModel/keggle/best_fundus_efficientnetb3.pth'
CLASS_NAMES = sorted([
    '0.0.Normal', '0.1.Tessellated fundus', '0.2.Large optic cup', '0.3.DR1', '1.0.DR2', '1.1.DR3',
    '10.0.Possible glaucoma', '10.1.Optic atrophy', '11.Severe hypertensive retinopathy',
    '12.Disc swelling and elevation', '13.Dragged Disc', '14.Congenital disc abnormality',
    '15.0.Retinitis pigmentosa', '15.1.Bietti crystalline dystrophy',
    '16.Peripheral retinal degeneration and break', '17.Myelinated nerve fiber', '18.Vitreous particles',
    '19.Fundus neoplasm', '2.0.BRVO', '2.1.CRVO', '20.Massive hard exudates',
    '21.Yellow-white spots-flecks', '22.Cotton-wool spots', '23.Vessel tortuosity',
    '24.Chorioretinal atrophy-coloboma', '25.Preretinal hemorrhage', '26.Fibrosis', '27.Laser Spots',
    '28.Silicon oil in eye', '29.0.Blur fundus without PDR', '29.1.Blur fundus with suspected PDR',
    '3.RAO', '4.Rhegmatogenous RD', '5.0.CSCR', '5.1.VKH disease', '6.Maculopathy', '7.ERM', '8.MH',
    '9.Pathological myopia', '1000images'
])

# -------- TRANSFORM --------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# -------- LOAD MODEL --------
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- INIT GRADCAM --------
cam_extractor = GradCAM(model, target_layer="conv_head")  # for EfficientNet

# -------- PREDICT & VISUALIZE FUNCTION --------
def predict_image_with_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    # DO NOT disable gradients — required for GradCAM
    output = model(input_tensor)
    probs = torch.sigmoid(output).cpu().detach().numpy()[0]

    # Print class probabilities
    print("/nClass Probabilities:")
    for i, prob in enumerate(probs):
        print(f"{CLASS_NAMES[i]}: {prob:.4f}")

    predicted_idx = int(np.argmax(probs))
    print(f"/n✅ Predicted Class: {CLASS_NAMES[predicted_idx]}")

    # Generate Grad-CAM
    cam_map = cam_extractor(predicted_idx, output)[0].cpu()

    # Denormalize for display
    img_vis = input_tensor.squeeze().cpu()
    img_vis = (img_vis * 0.5) + 0.5
    img_pil = to_pil_image(img_vis)

    # Overlay Grad-CAM on image
    cam_result = overlay_mask(img_pil, to_pil_image(cam_map, mode='F'), alpha=0.5)

    # Display result
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_result)
    plt.title(f"Grad-CAM: {CLASS_NAMES[predicted_idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -------- RUN TEST --------
test_image_path = r'D:/Kaif/Hackathon25/CDAC/1000images/5.1.VKH disease/1ffa93a4-8d87-11e8-9daf-6045cb817f5b..JPG'
predict_image_with_gradcam(test_image_path)
