import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes

# === Configuration ===
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
image_dir = "../../data/train512reduce4"
save_mask_dir = "../../data/sam_pseudo_masks"
os.makedirs(save_mask_dir, exist_ok=True)

# === Initialize SAM ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def clean_blobs(blobs, min_distance=15):
    """
    Clean blobs by performing non-maximum suppression based on a minimum distance threshold.
    Blobs is an array of shape (n, 3) where each row is (y, x, sigma).
    We sort by sigma (largest first) and remove blobs that are closer than min_distance.
    """
    if len(blobs) == 0:
        return blobs
    # Sort blobs descending by sigma (assumed proxy for blob size)
    sorted_idx = np.argsort(-blobs[:, 2])
    blobs_sorted = blobs[sorted_idx]
    cleaned = []
    for blob in blobs_sorted:
        y, x, sigma = blob
        # Only add the blob if it is not too close to any already selected blob
        if all(np.sqrt((y - b[0])**2 + (x - b[1])**2) >= min_distance for b in cleaned):
            cleaned.append(blob)
    return np.array(cleaned)

# === Process Images ===
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.tif')]

for fname in image_files:
    path = os.path.join(image_dir, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"[WARNING] Unable to read {fname}")
        continue

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    h, w, _ = img.shape

    # Step 1: Detect blobs (possible glomeruli)
    gray = rgb2gray(img)
    blobs = blob_log(gray, min_sigma=5, max_sigma=30, num_sigma=10, threshold=0.02)
    # Clean blob detections using non-maximum suppression
    blobs_clean = clean_blobs(blobs, min_distance=15)

    # Each blob is (y, x, sigma). If no blobs found, use the image center as fallback.
    if len(blobs_clean) == 0:
        print(f"[WARNING] No blobs found in {fname}, using center fallback")
        input_points = np.array([[w // 2, h // 2]])
    else:
        # Convert (y, x) -> (x, y) and round to nearest int
        input_points = np.round(blobs_clean[:, :2][:, ::-1]).astype(int)

    # All input points are considered positive prompts (label=1)
    input_labels = np.ones(len(input_points), dtype=int)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    # Step 2: Merge masks into one binary mask
    final_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # Step 3: Clean the final mask using morphological operations
    # Convert to boolean for processing
    final_mask_bool = final_mask.astype(bool)
    # Remove small objects and fill small holes
    cleaned_mask = remove_small_objects(final_mask_bool, min_size=50)
    cleaned_mask = remove_small_holes(cleaned_mask, area_threshold=50)
    # Convert back to uint8
    cleaned_mask = (cleaned_mask.astype(np.uint8)) * 255

    out_path = os.path.join(save_mask_dir, fname.replace('.tif', '.png'))
    cv2.imwrite(out_path, cleaned_mask)
    print(f"[INFO] Saved cleaned pseudo-mask: {out_path}, blobs used: {len(input_points)}")

print("✅ All pseudo masks generated and cleaned using blob detection and morphological operations.")
