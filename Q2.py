import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def color_region_grow(img, seed_point, threshold=30):
    h, w, _ = img.shape
    segmented = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    seed_color = img[seed_point[1], seed_point[0], :].astype(np.int32)

    queue = deque([seed_point])

    while queue:
        x, y = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True
        pixel_color = img[y, x, :].astype(np.int32)
        diff = np.linalg.norm(pixel_color - seed_color)

        if diff <= threshold:
            segmented[y, x] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        queue.append((nx, ny))
    return segmented

def main():
    image = cv2.imread('image.jpg')
    if image is None:
        return

    seed_point = (100, 100)
    result = color_region_grow(image, seed_point, threshold=35)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.scatter(*seed_point, color='red', s=30, label="Seed")
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title("Region Grown Output")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
