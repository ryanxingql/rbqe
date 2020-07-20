import numpy as np
import argparse
import matplotlib.pyplot as plt


nfs_test = 5

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type_test', type=str, default="HEVC", help="HEVC or JPEG")
    args = parser.parse_args()
    opt = args.type_test

    if opt == "HEVC":
        TQ = 0.84
        extract_dpsnr_HEVC()
        extract_QualityScore_HEVC()
        dpsnr_1000img_5QP_output = np.load("out/dpsnr_HEVC.npy")  # (nfs_test,5,6), 5 QP, psnr_ori and dpsnr of 5 outputs
        quality_1000img_5QP_output = np.load("out/QualityScore_HEVC.npy")  # (nfs_test,5,6), 5 QP, 5 outputs + cmp
        QX_list = [22, 27, 32, 37, 42]

    elif opt == "JPEG":
        TQ = 0.67
        extract_dpsnr_JPEG()
        extract_QualityScore_JPEG()
        dpsnr_1000img_5QP_output = np.load("out/dpsnr_JPEG.npy")  # (nfs_test,5,6), 5 QP, psnr_ori and dpsnr of 5 outputs
        quality_1000img_5QP_output = np.load("out/QualityScore_JPEG.npy")  # (nfs_test,5,6), 5 QP, 5 outputs + cmp
        QX_list = [50, 40, 30, 20, 10]

    #dpsnr_compared_1000img_5QP = np.load("out/DPSNR_other_list.npy") # (1000,5,4), 5 QP, 4 compared approaches
 
    FLOPs_list = [10.2, 14.7, 18.7, 22.9, 27.6]  # flops of each output. measured by thop: https://github.com/Lyken17/pytorch-OpCounter

    """
    # Validate the data, which should be the same as output/result_ave_HEVC.txt
    print("=== dpsnr validation ===")
    print("should be the same as output/result_ave_HEVC.txt")
    sum_ave = 0
    for ite_QP in range(5):
        sum_dpsnr = 0
        for ite_img in range(nfs_test):
            sum_dpsnr += dpsnr_1000img_5QP_output[ite_img, ite_QP, ite_QP]
        print("%d: %.3f" % (QX_list[ite_QP], sum_dpsnr / nfs_test))
        sum_ave += sum_dpsnr / nfs_test
    print("ave: %.3f" % (sum_ave / 5))
    """

    print("\n=== EXP 1: Ablation ===")
    # e.g.: we force all QP=22 images to output at output 1.
    for ite_QP in range(5):
        for ite_out in range(5):
            dpsnr_sum = 0
            FLOPs_sum = 0
            for ite_img in range(nfs_test): # test 1000 imgs

                dpsnr_sum += dpsnr_1000img_5QP_output[ite_img, ite_QP, ite_out]
                FLOPs_sum += FLOPs_list[ite_out]

            print("force all QP/QF=%d imgs to output at output=%d: dpsnr=%.3f, flops=%.3f" % (QX_list[ite_QP], ite_out + 1,\
                dpsnr_sum/nfs_test, FLOPs_sum /nfs_test))
    
    print("\n=== EXP 2: Tradeoff curve ===")
    print("observe the turning point and choose optimal T.")
    print("warning: we have only 5 images here, so the curve may not smooth.")
    dpsnr_list = []
    flops_list = []
    for TQ_test in np.arange(0, 1, 0.005):

        # See dPSNR-FLOPs under different TQ
        dpsnr_sum = 0
        FLOPs_sum = 0
        for ite_img in range(nfs_test): # test 1000 imgs
            for ite_QP in range(5): # test 5 imgs

                flag = 0
                for ite_out in range(4): # former 4 outputs
                    if quality_1000img_5QP_output[ite_img, ite_QP, ite_out] >= TQ_test:
                        dpsnr_sum += dpsnr_1000img_5QP_output[ite_img, ite_QP, ite_out]
                        FLOPs_sum += FLOPs_list[ite_out]
                        flag = 1
                        break

                if flag == 0: # the last output
                    dpsnr_sum += dpsnr_1000img_5QP_output[ite_img, ite_QP, 4]
                    FLOPs_sum += FLOPs_list[4]

        dpsnr_ave = dpsnr_sum / nfs_test / 5
        flop_ave = FLOPs_sum / nfs_test / 5
        dpsnr_list.append(dpsnr_ave)
        flops_list.append(flop_ave)
        print("%.3f: %.6f - %.6f" % (TQ_test, dpsnr_ave, flop_ave))
    plt.plot(flops_list, dpsnr_list)
    plt.xlabel('flops')
    plt.ylabel('dpsnr')
    plt.show()

    print("\n=== EXP 3: Optimal result ===")
    print("warning: we have only 5 images here, so the result may be strange.")
    print("chosen T: %.3f" % TQ)
    dpsnr_ave_sum = 0
    FLOPs_ave_sum = 0
    for ite_QP in range(5):
        dpsnr_sum = 0
        FLOPs_sum = 0
        for ite_img in range(nfs_test): # test 1000 imgs

            flag = 0
            for ite_out in range(4): # former 4 outputs
                if quality_1000img_5QP_output[ite_img, ite_QP, ite_out] >= TQ:
                    dpsnr_sum += dpsnr_1000img_5QP_output[ite_img, ite_QP, ite_out]
                    FLOPs_sum += FLOPs_list[ite_out]
                    flag = 1
                    break

            if flag == 0: # the last output
                dpsnr_sum += dpsnr_1000img_5QP_output[ite_img, ite_QP, 4]
                FLOPs_sum += FLOPs_list[4]

        dpsnr_ave_sum += dpsnr_sum / nfs_test
        FLOPs_ave_sum += FLOPs_sum / nfs_test

        print("QP/QF %d - %.6f - %.6f" % (QX_list[ite_QP], dpsnr_sum/nfs_test, FLOPs_sum /nfs_test))
    print("ave %.6f - %.6f" % (dpsnr_ave_sum / 5, FLOPs_ave_sum / 5))


def extract_dpsnr_HEVC():

    QP_list = [22, 27, 32, 37, 42]

    fp = open("out/result_EachFrame_HEVC.txt", 'r')
    dpsnr_array = np.zeros((nfs_test, 5, 6))
    while True:
        line = fp.readline()
        if not line:
            break

        pos_psnr = line.find("psnr")
        if pos_psnr < 0:
            continue

        pos_dpsnr = line.find("dpsnr")
        psnr_ori = float(line[pos_psnr+5:pos_dpsnr-2])

        pos_ = line.find("-")
        ite_frame = int(line[5:pos_])

        pos_2 = line.find("-", pos_ + 1)
        QP = int(line[pos_+4:pos_2])

        pos_o5 = line.find("o5")
        pos_c = line.find(",")
        dpsnr1 = float(line[pos_o5 + 3: pos_c])

        pos_c2 = line.find(",", pos_c + 1)
        dpsnr2 = float(line[pos_c + 1: pos_c2])

        pos_c3 = line.find(",", pos_c2 + 1)
        dpsnr3 = float(line[pos_c2 + 1: pos_c3])

        pos_c4 = line.find(",", pos_c3 + 1)
        dpsnr4 = float(line[pos_c3 + 1: pos_c4])

        dpsnr5 = float(line[pos_c4 + 1: ])

        dpsnr_array[ite_frame, QP_list.index(QP), 0] = dpsnr1
        dpsnr_array[ite_frame, QP_list.index(QP), 1] = dpsnr2
        dpsnr_array[ite_frame, QP_list.index(QP), 2] = dpsnr3
        dpsnr_array[ite_frame, QP_list.index(QP), 3] = dpsnr4
        dpsnr_array[ite_frame, QP_list.index(QP), 4] = dpsnr5
        dpsnr_array[ite_frame, QP_list.index(QP), 5] = psnr_ori

    np.save("out/dpsnr_HEVC.npy", dpsnr_array)
    fp.close()


def extract_dpsnr_JPEG():

    QF_list = [50, 40, 30, 20, 10]

    fp = open("out/result_EachFrame_JPEG.txt", "r")
    dpsnr_array = np.zeros((nfs_test, 5, 6))
    while True:
        line = fp.readline()
        if not line:
            break

        pos_psnr = line.find("psnr")
        if pos_psnr < 0:
            continue

        pos_dpsnr = line.find("dpsnr")
        psnr_ori = float(line[pos_psnr+5:pos_dpsnr-2])

        pos_ = line.find("-")
        ite_frame = int(line[5:pos_])

        pos_2 = line.find("-", pos_ + 1)
        QF = int(line[pos_+4:pos_2])

        pos_o5 = line.find("o5")
        pos_c = line.find(",")
        dpsnr1 = float(line[pos_o5 + 3: pos_c])

        pos_c2 = line.find(",", pos_c + 1)
        dpsnr2 = float(line[pos_c + 1: pos_c2])

        pos_c3 = line.find(",", pos_c2 + 1)
        dpsnr3 = float(line[pos_c2 + 1: pos_c3])

        pos_c4 = line.find(",", pos_c3 + 1)
        dpsnr4 = float(line[pos_c3 + 1: pos_c4])

        dpsnr5 = float(line[pos_c4 + 1: ])

        dpsnr_array[ite_frame, QF_list.index(QF), 0] = dpsnr1
        dpsnr_array[ite_frame, QF_list.index(QF), 1] = dpsnr2
        dpsnr_array[ite_frame, QF_list.index(QF), 2] = dpsnr3
        dpsnr_array[ite_frame, QF_list.index(QF), 3] = dpsnr4
        dpsnr_array[ite_frame, QF_list.index(QF), 4] = dpsnr5
        dpsnr_array[ite_frame, QF_list.index(QF), 5] = psnr_ori

    np.save("out/dpsnr_JPEG.npy", dpsnr_array)
    fp.close()


def extract_QualityScore_HEVC():

    fp = open("out/report_QualityScore_HEVC.txt", 'r')
    QP_list = [22, 27, 32, 37, 42]
    results_array = np.zeros((nfs_test, 5, 6), dtype=np.float32)
    while True:
        line = fp.readline()
        if not line:
            break

        pos_ = line.find("-")
        ite_img = int(line[0:pos_]) - 1

        pos_QP = line.find("QP")
        QP = int(line[pos_QP + 2:pos_QP + 5])
        ite_QP = QP_list.index(QP)

        pos_out = line.find("output")
        pos_ = line.find("-", pos_out)
        out = line[pos_out + 6:pos_].strip()
        if out == "cmp":
            ite_out = 5
        else:
            ite_out = int(out) - 1

        quality = float(line[pos_+1:])
        results_array[ite_img, ite_QP, ite_out] = quality

    np.save("out/QualityScore_HEVC.npy", results_array)
    fp.close()


def extract_QualityScore_JPEG():

    fp = open("out/report_QualityScore_JPEG.txt", "r")
    QF_list = [50, 40, 30, 20, 10]
    results_array = np.zeros((nfs_test, 5, 6), dtype=np.float32)
    while True:
        line = fp.readline()
        if not line:
            break

        pos_ = line.find("-")
        ite_img = int(line[0:pos_]) - 1

        pos_QF = line.find("QF")
        QF = int(line[pos_QF + 2:pos_QF + 5])
        ite_QF = QF_list.index(QF)

        pos_out = line.find("output")
        pos_ = line.find("-", pos_out)
        out = line[pos_out + 6:pos_].strip()
        if out == "cmp":
            ite_out = 5
        else:
            ite_out = int(out) - 1

        quality = float(line[pos_+1:])
        results_array[ite_img, ite_QF, ite_out] = quality

    np.save("out/QualityScore_JPEG.npy", results_array)
    fp.close()


if __name__ == "__main__":
    main()