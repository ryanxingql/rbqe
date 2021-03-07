"""
python -m pip install opencv-python
"""
import os
import torch
import numpy as np
from glob import glob
from cv2 import cv2
from network import network

gpu = '0'
cwd = os.getcwd()
dir_src = os.path.join(cwd, '../kodak/jpeg/qf20')
dir_tar = os.path.join(cwd, '../tmp')
if not os.path.exists(dir_tar):
    os.makedirs(dir_tar)

model_path = 'model/model_JPEG.pt'

dev = torch.device("cuda:" + gpu)
im_lst = glob(os.path.join(dir_src, '*.png'))

mat1 = np.array([
    [  65.481,  128.553,   24.966],
    [ -37.797,  -74.203,  112.   ],
    [ 112.   ,  -93.786,  -18.214],
    ]) / 225.

mat2 = np.linalg.inv(mat1)

def _rgb2ycbcr(im):
    """
    https://github.com/RyanXingQL/PythonUtils/blob/4d76981852106f9fda209eb34285eb303c72e412/conversion.py#L94
    """
    im = im.copy()
    im = im.astype(np.float64)
    
    im = im.dot(mat1.T)
    im[:,:,0] += 16.
    im[:,:,[1,2]] += 128.
    
    def _uint8(im):
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    im = _uint8(im)
    return im

def _tensor2im(t):
    """
    https://github.com/RyanXingQL/PythonUtils/blob/4d76981852106f9fda209eb34285eb303c72e412/conversion.py#L19
    """
    im = t.cpu().detach().numpy()  # as copy in numpy

    def _float2uint8(im):
        im *= 255.
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    
    im = _float2uint8(im)
    return im

def _ycbcr2rgb(im):
    """YCbCr -> RGB. 444P.
    Input: (H W C) uint8 image.
    
    Y is in the range [16,235]. Yb and Cr are in the range [16,240].
    See: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    """
    im = im.copy()
    im = im.astype(np.float64)
    
    im[:,:,0] -= 16.
    im[:,:,[1,2]] -= 128.

    """
    mat = np.array([
        [255./219., 0., 255./224.],
        [255./219., -255./224.*1.772*0.114/0.587, -255./224.*1.402*0.299/0.587],
        [255./219., 255./224.*1.772, 0.],
        ])  # actually the inverse matrix of (that in rgb2ycbcr / 255.)
    """
    im = im.dot(mat2.T)  # error when using mat2 is smaller
    
    def _uint8(im):
        im = im.round()  # first round. directly astype will cut decimals
        im = im.clip(0, 255)  # else, -1 -> 255, -2 -> 254!
        im = im.astype(np.uint8)
        return im
    im = _uint8(im)
    return im

def main():
    model = network()
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()
    print('> successfully restored ' + model_path)

    with torch.no_grad():
        test(model)
    print('> finish.')

def test(model):
    for im_path in im_lst:
        bgr_im = cv2.imread(im_path)  # H W BGR
        ycbcr_im = _rgb2ycbcr(bgr_im[:,:,::-1])  # H W RGB -> H W YCbCr
        y_im = np.squeeze(ycbcr_im[:,:,0])
        y_im = y_im.astype(np.float32) / 255.
        y_t = torch.from_numpy(y_im[np.newaxis, np.newaxis, :, :]).to(dev)
        
        im_name = im_path.split('/')[-1].split('.')[0]

        for idx_out in range(1, 6):
            enh_y = _tensor2im(model(y_t, idx_out))
            enh_ycbcr_im = ycbcr_im.copy()
            enh_ycbcr_im[:,:,0] = enh_y
            bgr_im = _ycbcr2rgb(enh_ycbcr_im)[:,:,::-1]
            tar_path = os.path.join(dir_tar, im_name + '-' + str(idx_out) + '.png')
            cv2.imwrite(tar_path, bgr_im)
            print(tar_path)

if __name__ == '__main__':
    main()