from pathlib import Path
import cv2
from src.registration import ImageRegistration
from utils.tools import checkerboard
from settings import NUM_TILES

REFERENCE_PATH = Path("img/GRF_RGB_revised.tiff")
MOVING_PATH = Path("img/GRF_RGB_U_revised.tiff")
OUTPUT_PATH = Path("img")


def crop_to_bbox(image, bbox):
    y0, x0, y1, x1 = bbox
    return image[y0:y1, x0:x1]


if __name__ == "__main__":
    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    moving = cv2.imread(str(MOVING_PATH), cv2.IMREAD_COLOR)

    image_registration = ImageRegistration()
    refined_aligned_full, refined_mask_full, homography, info, bbox = image_registration.register(reference, moving)

    cv2.imwrite(str(OUTPUT_PATH / "GRF_aligned_u.tiff"), refined_aligned_full)
    cv2.imwrite(str(OUTPUT_PATH / "GRF_aligned_u_mask.tiff"), refined_mask_full)

    checker = checkerboard(crop_to_bbox(reference, bbox), crop_to_bbox(refined_aligned_full, bbox), num_tiles=NUM_TILES)
    cv2.imwrite(str(OUTPUT_PATH / "GRF_checkerboard.tiff"), checker)

    print(f"homography:\n{homography}")
    print(f"bbox:\n{bbox}")
    print(info)
