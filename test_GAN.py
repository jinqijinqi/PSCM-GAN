import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from scipy.misc import imread, imsave
from imageio import imread, imsave
from glob import glob
from MODEL2 import CGAN  # Assuming CGAN class is in MODEL2.py

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def prepare_data(dataset):
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob(os.path.join(data_dir, "*.jpg")))
    return data

def lrelu(x, leak=0.2):
    return torch.maximum(x, leak * x)

def imread(path, is_grayscale=True):
    if is_grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)

def imsave(image, path):
    cv2.imwrite(path, image)

def input_setup(index, data_ir, data_vi, data_ref):
    padding = 0
    sub_ir_sequence = []
    sub_vi_sequence = []
    sub_ref_sequence = []

    input_ir = cv2.resize((imread(data_ir[index]) - 127.5) / 127.5, (456, 456))
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    input_ir = input_ir.reshape((input_ir.shape[0], input_ir.shape[1], 1))
    sub_ir_sequence.append(input_ir)

    input_vi = cv2.resize((imread(data_vi[index]) - 127.5) / 127.5, (456, 456))
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    input_vi = input_vi.reshape((input_vi.shape[0], input_vi.shape[1], 1))
    sub_vi_sequence.append(input_vi)

    input_ref = cv2.resize((imread(data_ref[index]) - 127.5) / 127.5, (456, 456))
    input_ref = np.pad(input_ref, ((padding, padding), (padding, padding)), 'edge')
    input_ref = input_ref.reshape((input_ref.shape[0], input_ref.shape[1], 1))
    sub_ref_sequence.append(input_ref)

    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    train_data_ref = np.asarray(sub_ref_sequence)

    return train_data_ir, train_data_vi, train_data_ref

def main():
    num_epoch = 27
    path = '_100_onlyadd_THREE22'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while num_epoch == 27:
        print("num_epoch", num_epoch)
        #checkpoint_path = f'./checkpoint_20/CGAN{path}/CGAN.model-{num_epoch}.pth'
        #checkpoint_path = f'./checkpoint_20/model/model_epoch1_step4000.pth'
        #print(checkpoint_path)

        # Load the model
        #model = CGAN()
        #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model=torch.jit.load('./checkpoint_20/model-git/model.pt')
        model.to(device)
        model.eval()

        data_ir = prepare_data('dataset/IR')
        data_vi = prepare_data('dataset/VIS')
        data_ref = prepare_data('dataset/IR') # for test case

        for i in range(len(data_ir)):
            start = time.time()
            train_data_ir, train_data_vi, train_data_ref = input_setup(i, data_ir, data_vi, data_ref)

            with torch.no_grad():
                train_data_ir = torch.tensor(train_data_ir, device=device).permute(0, 3, 1, 2)
                train_data_vi = torch.tensor(train_data_vi, device=device).permute(0, 3, 1, 2)
                train_data_ref = torch.tensor(train_data_ref, device=device).permute(0, 3, 1, 2)

                fusion_image = model(train_data_ir, train_data_vi)
                fusion_image = fusion_image * 127.5 + 127.5
                result = fusion_image.squeeze().cpu().numpy()

               # image_path = f'./with_enh_VOT_Result/epoch{num_epoch}'
                image_path = f'./result'
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                image_path = os.path.join(image_path, f"{i + 1}.bmp")

                imsave(result, image_path)
                end = time.time()
                print(f"Testing [{i}] success, Testing time is [{end - start}]")

        num_epoch += 1
        break

if __name__ == '__main__':
    main()
