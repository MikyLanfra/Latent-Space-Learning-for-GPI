from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib.pyplot as plt

def find_blue(img):
    blue_mask = (img[:,:,0] < 150) & (img[:,:,1] < 150) & (img[:,:,2] > 200)
    return blue_mask

def find_gray(img):
    grey_mask = (img[:,:,0] < 200) & (img[:,:,1] < 200) & (img[:,:,2] < 200)
    return grey_mask

def plot_dot_trajectory(images):
    x_arr, y_arr = [], []
    for img in images:
        blue_mask = find_blue(img)
        if blue_mask.any():
            y, x = np.where(blue_mask)
            center_x = int(np.mean(x))
            center_y = int(np.mean(y))
            x_arr.append(center_x)
            y_arr.append(center_y)
    plt.figure(figsize=(4,4))
    plt.plot(x_arr, y_arr, marker='o')
    plt.xlim(0,96)
    plt.ylim(0,96)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()
    return np.array(x_arr), np.array(y_arr)

def move_blue_dot(imgs, offset=50, save=True):
    random_idx = np.random.randint(0, len(imgs))
    img = imgs[random_idx]
    b_mask = find_blue(img)
    g_mask = find_gray(img)
    left = np.min(np.where(b_mask)[1])
    right = np.max(np.where(b_mask)[1])
    top = np.min(np.where(b_mask)[0])
    bottom = np.max(np.where(b_mask)[0])
    # print(f"Blue region - Left: {left}, Right: {right}, Top: {top}, Bottom: {bottom}")
    img_mod = img.copy()
    dot_copy = img_mod[top:bottom+1, left:right+1, :].copy()
    img_mod[top:bottom+1, left:right+1, :] = 255

    while True:
        new_x = np.random.randint(-offset, offset)
        new_y = np.random.randint(-offset, offset)
        new_mask = np.zeros_like(b_mask)
        new_mask[top+new_y:bottom+1+new_y, left+new_x:right+1+new_x] = True
        if not g_mask[new_mask].any():
            try:
                img_mod[top+new_y:bottom+1+new_y, left+new_x:right+1+new_x, :] = dot_copy
            except ValueError: # If out of bounds, retry
                continue 
            break

    plt.imshow(img_mod/255)
    plt.show()

    # print(img_mod.shape)
    # Save modified image
    
    if save: 
        name = "modified_image_" + str(random_idx) + ".png"
        save_image(torch.tensor(img_mod).permute(2,0,1)/255, name)
    return torch.tensor(img_mod), random_idx