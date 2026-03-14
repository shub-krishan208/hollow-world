import numpy as np
import json
import cv2
import os

def extrapolate():
    capture_dir = r"C:\Users\Yashi\My project (1)\Assets\depth_capture"
    
    depth_path = os.path.join(capture_dir, "depth_map.png")
    meta_path = os.path.join(capture_dir, "meta.json")
    output_path = os.path.join(capture_dir, "points.json")

    # Check if files exist before starting
    if not os.path.exists(depth_path) or not os.path.exists(meta_path):
        print(f"Error: Missing files in {capture_dir}")
        return
    # 1. Load Depth Map (Assume 0-255 grayscale image from your AI)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth_in_meters = (depth_img / 255.0) * 10.0 # Adjust '10.0' to your AI's max range

    # 2. Load Unity Metadata
    with open("Captures/meta.json", "r") as f:
        meta = json.load(f)
    
    # 3. Reconstruct Pose Matrix
    m = meta['camToWorld']
    pose = np.array([
        [m['m00'], m['m01'], m['m02'], m['m03']],
        [m['m10'], m['m11'], m['m12'], m['m13']],
        [m['m20'], m['m21'], m['m22'], m['m23']],
        [m['m30'], m['m31'], m['m32'], m['m33']]
    ])

    world_points = []
    step = 10 # Only take every 10th pixel to save memory

    for v in range(0, depth_img.shape[0], step):
        for u in range(0, depth_img.shape[1], step):
            z = depth_in_meters[v, u]
            if z < 0.5: continue # Ignore things too close/background

            # THE MATH: Back-projection
            x_c = (u - meta['cx']) * z / meta['fx']
            y_c = -(v - meta['cy']) * z / meta['fy'] # Flip Y for Unity
            z_c = z

            # Transform to World Space
            p_cam = np.array([x_c, y_c, z_c, 1.0])
            p_world = pose @ p_cam
            
            world_points.append({"x": p_world[0], "y": p_world[1], "z": p_world[2]})

    # 4. Save for Unity
    with open("Captures/points.json", "w") as f:
        json.dump({"material": "Granite", "points": world_points}, f)

extrapolate()