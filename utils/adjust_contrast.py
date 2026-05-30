import cv2
import tifffile as tiff


def adjust_contrast(image, alpha=2.0, beta=0):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


if __name__ == "__main__":
    input_path = "hsi_rgb/GRF_RGB.tiff"
    output_path = "hsi_rgb/GRF_RGB_revised.tiff"

    image = tiff.imread(input_path)

    adjusted_image = adjust_contrast(image, alpha=2.0, beta=0)

    tiff.imwrite(output_path, adjusted_image)

    print(f"Saved: {output_path}")
