import torch
import cv2
import os
import numpy as np

# 1. Your Specific Windows Paths
INPUT_FOLDER = "/Users/shub/rack/code-rack/playground/hollow/input"
OUTPUT_FOLDER = "/Users/shub/rack/code-rack/playground/hollow/output"


# 2. Diagnostic Check (To see why it might be skipping files)
if not os.path.exists(INPUT_FOLDER):
    print(f"ERROR: The input folder was not found at {INPUT_FOLDER}")
else:
    all_files = os.listdir(INPUT_FOLDER)
    print(f"Diagnostic: Found {len(all_files)} total files in the input folder.")
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', 'heic'))]
    print(f"Diagnostic: Found {len(image_files)} compatible images to process.")

# 3. Load Model
# Note: This will finish the 1.28GB download you saw in your screenshot
#model_type = "MiDaS_small" 
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# 4. Move model to GPU (NVIDIA CUDA) or use mps
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

midas.to(device)
midas.eval()

# 5. Load Transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Device:", device)

# 6. Process Folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {filename}: Could not read image file.")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform for model
        input_batch = transform(img_rgb).to(device)

        # Prediction
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # 7. Normalize to 0-255 (Black & White: White=Close, Black=Far)
        output_bw = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 8. Save result
        output_path = os.path.join(OUTPUT_FOLDER, f"depth_{filename}")
        cv2.imwrite(output_path, output_bw)
        print(f"Done: {filename}")

print("\nFinished! Your black and white depth maps are ready.")
