import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_files(dataroot):
    files_list = []
    for root, dirs, files in os.walk(dataroot):
        for file in files:
            if file.endswith('.mp4'):
                files_list.append(os.path.join(root, file))
    return files_list

def center_crop(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

class DAVISDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        image_size,
        num_frames = 16,
        frame_skip = 1):

        self.root_dir = root_dir
        self.video_paths = get_files(self.root_dir)

        self.image_size = image_size
        self.num_frames = num_frames
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        tensor = video_to_tensor(video_path, self.image_size, self.num_frames, self.frame_skip)
        return tensor

def video_to_tensor(path, image_size=256):

    cap = cv2.VideoCapture(str(path))
    frame_list = []

    while True:
        ret, frame = cap.read()
        if (ret == True):
            frame = center_crop(frame, 480)
            resized_frame = cv2.resize(frame, (image_size, image_size))
            frame_list.append(torch.from_numpy(resized_frame).unsqueeze(0)/255.0)
        else:
            break

    cap.release()

    video = torch.cat(frame_list, dim=0)
    tensor = video.permute(3,0,1,2).contiguous().type(torch.float32)
    tensor = torch.flip(tensor, dims=(0,))

    return tensor

# Define the directory
data_dir = 'DAVIS_DATA_PATH'
dataset = DAVISDataset(root_dir=f'{data_dir}', image_size = 256, num_frames = 16, frame_skip = 1)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

dl = cycle(dataloader)

for i in range():
    images = next(dl)
    print(images.shape)
    print(images.shape[2])
    total_num = images.shape[2] // 16

    # Create a directory for the current tensor
    path = os.path.join('DATA_SAVE_PATH')
    if not os.path.exists(path):
        os.makedirs(path)

    for num in range(total_num):
        images_batch = images[0,:,num*16:(num+1)*16,:,:]

        # Convert the tensor to a NumPy array
        numpy_array = images_batch.numpy()

        split_file_name = os.path.join(path, f'split_{num+1:03d}.npy')

        # Save the NumPy array to a file
        np.save(split_file_name, numpy_array)