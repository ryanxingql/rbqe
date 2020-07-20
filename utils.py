import numpy as np
import math

def y_import(video_path, height_frame, width_frame, nfs, startfrm, opt_bar=False, opt_clear=False):
    """
    import Y channel from a yuv 420p video.
    startfrm: start from 0
    return: y_batch, (nfs * height * width), dtype=uint8
    """
    fp_data = open(video_path, 'rb')

    y_size = height_frame * width_frame
    u_size = height_frame // 2 * (width_frame // 2)
    v_size = u_size

    # target at startfrm
    blk_size = y_size + u_size + v_size
    fp_data.seek(blk_size * startfrm, 0)

    # extract
    y_batch = []
    for ite_frame in range(nfs):
        
        y_frame = [ord(fp_data.read(1)) for k in range(y_size)]
        y_frame = np.array(y_frame, dtype=np.uint8).reshape((height_frame, width_frame))
        fp_data.read(u_size + v_size)  # skip u and v
        y_batch.append(y_frame)

        if opt_bar:
            print("\r<%d, %d>" % (ite_frame, nfs - 1), end="", flush=True)
    if opt_clear:
        print("\r" + 20 * " ", end="\r", flush=True)

    fp_data.close()
    
    y_batch = np.array(y_batch)
    return y_batch


def cal_psnr(img1, img2, data_range=1.0):
    """
    calculate psnr of two imgs
    
    img1, img2: (C H W), [0, data_range]
    
    return ave of psnrs of all channels], np.float32
    """
    assert (len(img1.shape) == 3), "len(img1.shape) != 3!"
    assert (img1.shape == img2.shape), "img1.shape != img2.shape!"
    img1, img2 = _as_floats(img1, img2) # necessary!!!
    mse_channels = [cal_mse(img1[i], img2[i]) for i in range(img1.shape[0])]
    if min(mse_channels) == 0:
        return float('inf')
    psnr_channels = [10 * math.log10(float(data_range**2) / mse) for mse in mse_channels]
    return np.mean(psnr_channels, dtype=np.float32)


def cal_mse(img1, img2):
    """
    calculate mse (mean squared error) of two imgs.
    
    img1, img2: (H W)
    
    return mse, np.float32
    """
    assert (len(img1.shape) == 2), "len(img1.shape) != 2!"
    assert (img1.shape == img2.shape), "img1.shape != img2.shape!"
    img1, img2 = _as_floats(img1, img2) # necessary!!!
    return np.mean((img1 - img2)**2, dtype=np.float32) # default to average flattened array. so no need to reshape into 1D array


def _as_floats(img1, img2):
    """
    promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(img1.dtype, img2.dtype, np.float32)
    img1 = np.asarray(img1, dtype=float_type)
    img2 = np.asarray(img2, dtype=float_type)
    return img1, img2

