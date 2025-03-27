import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


base_folder = os.path.abspath("./project/predictions/emnist_analysis_reverse/p-i_t-l")
print(base_folder)
output_base = os.path.abspath("./project/predictions/emnist_reverse_plots/")

os.makedirs(output_base, exist_ok=True)


folder1 = "image_0"


def load_images_from_folder(folder_path, num_images=12):
    layer_path = os.path.join(folder_path, "layer_5")
    if not os.path.exists(layer_path):
        return []

    image_files = sorted(glob.glob(os.path.join(layer_path, "*.jpg")))[:num_images]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files if os.path.exists(f)]

    if len(images) < num_images:
        print(f"Warning: Expected {num_images} images, but found {len(images)} in {folder_path}")

    return images


def get_original_image(folder_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if image_files:
        image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        letter_name = os.path.basename(image_files[0]).split(".")[0]
        return image, letter_name
    return None, None


path1 = os.path.join(base_folder, folder1)

images1 = load_images_from_folder(path1)

num_channels = len(images1)
if num_channels == 0:
    print("No valid images found in one or both folders.")
    exit()


mean_activation_1 = [np.mean(img) for img in images1]

var_activation_1 = [np.var(img) for img in images1]


original_image_1, letter_1 = get_original_image(path1)

letter_1 = letter_1 if letter_1 else folder1
letter_1 = letter_1[-1]
letter_2 = 'f'

print(f"Comparing: {folder1} ({letter_1}))")

df_comparison = pd.DataFrame({
    "Channel": list(range(num_channels)),
    f"Mean Activation ({letter_1})": mean_activation_1[:num_channels],
    f"Variance ({letter_1})": var_activation_1[:num_channels],
})

comparison_folder = os.path.join(output_base, f"comparison_{folder1}")
os.makedirs(comparison_folder, exist_ok=True)

df_comparison.to_csv(os.path.join(comparison_folder, "activation_comparison.csv"), index=False)


plt.figure(figsize=(10, 5))
plt.plot(range(num_channels), mean_activation_1[:num_channels], marker='o', label=f"{letter_1} - Mean Activation")
plt.xlabel("Channel")
plt.ylabel("Mean Activation")
plt.title(f"Mean Activation: original {letter_1} predicted {letter_2} ")
plt.legend()
plt.savefig(os.path.join(comparison_folder, "mean_comparison.png"))
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(range(num_channels), var_activation_1[:num_channels], marker='o', label=f"{letter_1} - Variance")
plt.xlabel("Channel")
plt.ylabel("Variance")
plt.title(f"Variance: original {letter_1} predicted {letter_2}")
plt.legend()
plt.savefig(os.path.join(comparison_folder, "variance_comparison.png"))
plt.close()


if original_image_1 is not None:
    cv2.imwrite(os.path.join(comparison_folder, f"original_{folder1}.png"), original_image_1)

print(f"Comparison saved: {comparison_folder}")
