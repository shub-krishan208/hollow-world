import os
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor


def isolate_center_objects(input_folder, output_folder, checkpoint_path, model_type="vit_t"):
    print("Loading MobileSAM model...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = (".png", ".jpg", ".jpeg")

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith(valid_extensions):
            continue

        print(f"Processing: {filename}")

        filepath = os.path.join(input_folder, filename)

        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            predictor.set_image(image_rgb)

        height, width = image_rgb.shape[:2]
        center_x, center_y = width // 2, height // 2

        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])

        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        # ==========================================
        # Size-Filtered Confidence (same algorithm)
        # ==========================================

        total_image_area = height * width
        valid_masks = []

        for i in range(len(masks)):
            mask_area = np.sum(masks[i])
            area_ratio = mask_area / total_image_area

            if 0.02 < area_ratio < 0.75:
                valid_masks.append((masks[i], scores[i]))

        if len(valid_masks) > 0:
            valid_masks.sort(key=lambda x: x[1], reverse=True)
            best_mask = valid_masks[0][0]
        else:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

        # ==========================================

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[..., :3] = image_rgb
        rgba_image[..., 3] = best_mask * 255

        bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)

        output_filename = os.path.splitext(filename)[0] + "_isolated.png"
        output_filepath = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_filepath, bgra_image)

        print(f"Saved isolated object to: {output_filepath}")


if __name__ == "__main__":

    INPUT_DIR = "input"
    OUTPUT_DIR = "output"

    CHECKPOINT = "mobile_sam.pt"

    isolate_center_objects(INPUT_DIR, OUTPUT_DIR, CHECKPOINT, model_type="vit_t")
