import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_image():
    """
    Create a grayscale image with two rectangular objects and a dark background.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (35, 35), 90, -1)   # Object 1
    cv2.rectangle(img, (60, 60), (85, 85), 180, -1)  # Object 2
    return img

def apply_gaussian_noise(img, mean=0, stddev=25):
    """
    Add Gaussian noise to an image with specified mean and standard deviation.
    """
    noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy_img.astype(np.uint8)

def apply_otsu_thresholding(noisy_img):
    """
    Apply Otsu's method using OpenCV's built-in function.
    """
    _, thresholded = cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def display_results(original, noisy, thresholded):
    """
    Plot original, noisy, and thresholded images side-by-side.
    """
    titles = ['Original', 'With Gaussian Noise', "Otsu's Threshold"]
    images = [original, noisy, thresholded]
    
    plt.figure(figsize=(10, 3))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    base_img = generate_synthetic_image()
    noisy_img = apply_gaussian_noise(base_img)
    thresholded_img = apply_otsu_thresholding(noisy_img)
    display_results(base_img, noisy_img, thresholded_img)

if __name__ == "__main__":
    main()
