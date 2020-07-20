import os, time, glob, argparse
import numpy as np
import torch
from torch import nn

import utils

import imageio
#from skimage.metrics import structural_similarity as compare_ssim
#from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from network import network
#from thop import clever_format, profile  # for flops calculation: https://github.com/Lyken17/pytorch-OpCounter


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type_test', type=str, default="HEVC", help="HEVC or JPEG")
parser.add_argument('-g', '--gpu', type=str, default="0", help="GPU")
args = parser.parse_args()

dir_test = "data"
dir_model = "model"
dir_save = "out"
dir_save_img = os.path.join(dir_save, "img_" + args.type_test)

if not os.path.exists(dir_save):
    os.mkdir(dir_save)
txt_ave_path = os.path.join(dir_save, "result_ave_" + args.type_test + ".txt")
txt_dpsnr_path = os.path.join(dir_save, "result_EachFrame_" + args.type_test + ".txt")
fp_ave = open(txt_ave_path, 'w')
fp_each = open(txt_dpsnr_path, 'w')  # cal and record dpsnr and dssim of each img

dev = torch.device("cuda:" + args.gpu)
print(dev)

if args.type_test == "HEVC":
    order_QPorQF = [22,27,32,37,42]  # simpler the sample, earlier the output
    suffix_data_path = "512x512_test.yuv"
    tab = "qp"
    height_frame = 512
    width_frame = 512
elif args.type_test == "JPEG":
    order_QPorQF = [50,40,30,20,10]
    suffix_data_path = "512x512_test_jpeg.yuv"
    tab = "qf"
    height_frame = 512
    width_frame = 512

height_test = 512
width_test = 512
nfs_test_used = 5
start_height = height_frame // 2 - height_test // 2
start_width = width_frame // 2 - width_test // 2

opt_output = True  # save enhanced images
if opt_output:
    if not os.path.exists(dir_save_img):
        os.mkdir(dir_save_img)


def main():

    # restore model parameters and move to GPU
    model = network()
    
    # Cal num of paras
    # default: calculate the FLOPS of the latest output. See network.py order
    #input = torch.randn(1, 1, height_test, width_test)
    #flops, params = profile(model, inputs=(input, ))
    #flops, params = clever_format([flops, params], "%.3f")
    #print(flops)
    #print(params)

    # record time
    time_start = time.time()

    # restore model
    model_path = os.path.join(dir_model, "model_" + args.type_test + ".pt")
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()
    print("\n=== Successfully restore model! ===")
    fp_ave.write("\n=== Successfully restore model! ===\n")
    
    # test
    test(model)

    time_all = (time.time() - time_start) / 3600
    print("=== Time consuming: %.1f h ===" % time_all)
    fp_ave.write("=== Time consuming: %.1f h ===\n" % time_all)
    fp_ave.flush()
    
    fp_ave.close()


def test(model):

    with torch.no_grad():

        raw_path = os.path.join(dir_test, "RAISE_raw_" + suffix_data_path)
    
        dpsnr_sum_5QP = 0.0
        #dssim_sum_5QP = 0.0
        
        for QPorQF in order_QPorQF: # test order, not the output order
    
            cmp_path = os.path.join(dir_test, "RAISE_" + tab + str(QPorQF) + "_" + suffix_data_path)

            dpsnr_ave = 0.0
            #dssim_ave = 0.0
            time_total = 0.0
            nfs_test_final = nfs_test_used
            
            for ite_frame in range(nfs_test_used):

                raw_frame = utils.y_import(raw_path, height_frame, width_frame, nfs=1, startfrm=ite_frame).astype(np.float32)[:,start_height:start_height+height_test,start_width:start_width+width_test] / 255
                cmp_frame = utils.y_import(cmp_path, height_frame, width_frame, nfs=1, startfrm=ite_frame).astype(np.float32)[:,start_height:start_height+height_test,start_width:start_width+width_test] / 255
                if isplane(raw_frame): # plain frame => no need to enhance => invalid
                    nfs_test_final -= 1
                    continue
                
                cmp_t, raw_t = torch.from_numpy(cmp_frame).to(dev), torch.from_numpy(raw_frame).to(dev) # turn them to tensors and move to GPU
                cmp_t = cmp_t.view(1, 1, height_test, width_test) # batch_size * height * width => batch_size * channel * height * width
                
                start_time = time.time()
                
                enh_1 = model(cmp_t, 1) # enhanced img from the shallowest output
                enh_2 = model(cmp_t, 2)
                enh_3 = model(cmp_t, 3)
                enh_4 = model(cmp_t, 4)
                enh_5 = model(cmp_t, 5) # enhanced img from the deepest output
                
                if QPorQF == order_QPorQF[0]:
                    enhanced_cmp_t = enh_1
                elif QPorQF == order_QPorQF[1]:
                    enhanced_cmp_t = enh_2
                elif QPorQF == order_QPorQF[2]:
                    enhanced_cmp_t = enh_3
                elif QPorQF == order_QPorQF[3]:
                    enhanced_cmp_t = enh_4                    
                elif QPorQF == order_QPorQF[4]:
                    enhanced_cmp_t = enh_5
                    
                time_total += time.time() - start_time
                
                if opt_output: # save frame as png
                    func_output(ite_frame, QPorQF, cmp_t, out=0)
                    func_output(ite_frame, QPorQF, enh_1, out=1)
                    func_output(ite_frame, QPorQF, enh_2, out=2)
                    func_output(ite_frame, QPorQF, enh_3, out=3)
                    func_output(ite_frame, QPorQF, enh_4, out=4)
                    func_output(ite_frame, QPorQF, enh_5, out=5)                    
    
                # cal dpsnr and dssim
                #dpsnr, dssim = cal_dpsnr_dssim(raw_frame, cmp_frame, enhanced_cmp_t)
                dpsnr = cal_dpsnr_dssim(raw_frame, cmp_frame, enhanced_cmp_t)
                #print("\rframe %4d|%4d - dpsnr %.3f - dssim %3d (x1e-4) - %s %2d    " % (ite_frame + 1, nfs_test_used, dpsnr, dssim * 1e4, tab, QPorQF), end="", flush=True)
                print("\rframe %4d|%4d - dpsnr %.3f - %s %2d    " % (ite_frame + 1, nfs_test_used, dpsnr, tab, QPorQF), end="", flush=True)
                
                dpsnr_ave += dpsnr
                #dssim_ave += dssim
                
                # cal dpsnr and dssim for all outputs
                #dp1, ds1 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_1)
                #dp2, ds2 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_2)
                #dp3, ds3 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_3)
                #dp4, ds4 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_4)
                #dp5, ds5 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_5)
                dp1 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_1)
                dp2 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_2)
                dp3 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_3)
                dp4 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_4)
                dp5 = cal_dpsnr_dssim(raw_frame, cmp_frame, enh_5)
                
                
                fp_each.write("frame %d - %s %2d - ori psnr: %.3f - dpsnr from o1 to o5: %.3f, %.3f, %.3f, %.3f, %.3f\n" %\
                    (ite_frame, tab, QPorQF, utils.cal_psnr(cmp_frame, raw_frame, data_range=1.0), dp1, dp2, dp3, dp4, dp5))
                #fp_each.write("frame %d - %s %2d - ori ssim: %.3f - dssim from o1 to o5: %.3f, %.3f, %.3f, %.3f, %.3f\n" %\
                #    (ite_frame, tab, QPorQF, compare_ssim(np.squeeze(cmp_frame), np.squeeze(raw_frame), data_range=1), ds1, ds2, ds3, ds4, ds5))        
                fp_each.flush()
            
            dpsnr_ave = dpsnr_ave / nfs_test_final
            #dssim_ave = dssim_ave / nfs_test_final
            fps = nfs_test_final / time_total
            
            #print("\r=== dpsnr: %.3f - dssim %3d (x1e-4) - %s %2d - fps %.1f ===          " % (dpsnr_ave, dssim * 1e4, tab, QPorQF, fps), flush=True)
            print("\r=== dpsnr: {:.3f} - {:s} {:2d} - fps {:.1f} (no early-exit) ===".format(dpsnr_ave, tab, QPorQF, fps) + 10*" ", flush=True)
            #fp_ave.write("=== dpsnr: %.3f - dssim %3d (x1e-4) - %s %2d - fps %.1f ===\n" % (dpsnr_ave, dssim * 1e4, tab, QPorQF, fps))
            fp_ave.write("=== dpsnr: %.3f - %s %2d - fps %.1f (no early-exit) ===\n" % (dpsnr_ave, tab, QPorQF, fps))
            
            fp_ave.flush()
    
            dpsnr_sum_5QP += dpsnr_ave
            #dssim_sum_5QP += dssim_ave
            
        #print("=== dpsnr: %.3f - dssim: % 3d (x1e-4) ===" % (dpsnr_sum_5QP / 5, dssim_sum_5QP / 5 * 1e4), flush=True)
        #fp_ave.write("=== dpsnr: %.3f - dssim: % 3d (x1e-4) ===\n" % (dpsnr_sum_5QP / 5, dssim_sum_5QP / 5 * 1e4))
        print("=== dpsnr: %.3f ===" % (dpsnr_sum_5QP / 5, ), flush=True)
        fp_ave.write("=== dpsnr: %.3f ===\n" % (dpsnr_sum_5QP / 5))
        fp_ave.flush()


def cal_dpsnr_dssim(raw_frame, cmp_frame, enhanced_t):
    dpsnr = utils.cal_psnr(torch.squeeze(enhanced_t, 0).detach().cpu().numpy(), raw_frame, data_range=1.0) -\
        utils.cal_psnr(cmp_frame, raw_frame, data_range=1.0)
    #dpsnr = compare_psnr(torch.squeeze(enhanced_t).detach().cpu().numpy(), np.squeeze(raw_frame), data_range=1) -\
    #    compare_psnr(np.squeeze(cmp_frame), np.squeeze(raw_frame), data_range=1)
    #dssim = compare_ssim(torch.squeeze(enhanced_t).detach().cpu().numpy(), np.squeeze(raw_frame), data_range=1) -\
    #    compare_ssim(np.squeeze(cmp_frame), np.squeeze(raw_frame), data_range=1)
    return dpsnr#, dssim


def isplane(frame):
    """Detect black frames or other plane frames."""
    tmp_array = np.squeeze(frame).reshape([-1])

    if all(tmp_array[1:] == tmp_array[:-1]):  # all values in this frame are equal, e.g., black frame
        return True
    else:
        return False


def func_output(ite_frame, QPorQF, enhanced_cmp_t, out):
    if out == 0:
        output_path = os.path.join(dir_save_img, str(ite_frame) + "_" + tab + str(QPorQF) + "_cmp.bmp")
    else:
        output_path = os.path.join(dir_save_img, str(ite_frame) + "_" + tab + str(QPorQF) + "_out" + str(out) + ".bmp")
    imageio.imwrite(output_path, ((torch.squeeze(enhanced_cmp_t).detach().cpu().numpy()) * 255).astype(np.uint8))


if __name__ == '__main__':
    main()