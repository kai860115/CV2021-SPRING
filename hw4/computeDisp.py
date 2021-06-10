import numpy as np
import cv2
import time
from numpy.lib.arraypad import pad
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    window_size = 9
    padding_size = window_size // 2
    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    Il_pading = cv2.copyMakeBorder(Il_gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)
    Ir_pading = cv2.copyMakeBorder(Ir_gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    t0 = time.time()

    pattern_l = np.zeros((h, w, window_size * window_size, 1), dtype=np.bool)
    pattern_r = np.zeros((h, w, window_size * window_size, 1), dtype=np.bool)
    for i in range(h):
        for j in range(w):
            pattern_l[i, j] = (Il_pading[i: i+window_size, j: j+window_size] < Il_pading[i+padding_size, j+padding_size]).reshape(-1,1)
            pattern_r[i, j] = (Ir_pading[i: i+window_size, j: j+window_size] < Ir_pading[i+padding_size, j+padding_size]).reshape(-1,1)

    cost_l = np.ones((h, w, max_disp+1), dtype=np.float32) * (window_size ** 2)
    cost_r = np.ones((h, w, max_disp+1), dtype=np.float32) * (window_size ** 2)
    for i in range(h):
        for j in range(w):
            cost_l[i, j, :j+1] = np.logical_xor(pattern_l[i, j], pattern_r[i, j::-1])[:max_disp+1].sum(-1).sum(-1)
            cost_r[i, j, :w-j] = np.logical_xor(pattern_r[i, j], pattern_l[i, j:w:])[:max_disp+1].sum(-1).sum(-1)

    print('Cost Computation takes %.4f sec'%(time.time()-t0))

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    t0 = time.time()
    for shift in range(max_disp+1):
        cost_l[:, :, shift] = xip.jointBilateralFilter(Il_gray, cost_l[:, :, shift], 9, 0.9, 9)
        cost_r[:, :, shift] = xip.jointBilateralFilter(Ir_gray, cost_r[:, :, shift], 9, 0.9, 9)

    print('Cost Aggregation takes %.4f sec'%(time.time()-t0))

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    t0 = time.time()
    labels_l = np.argmin(cost_l, axis=-1)
    labels_r = np.argmin(cost_r, axis=-1)
    
    print('Disparity Optimization takes %.4f sec'%(time.time()-t0))

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    t0 = time.time()
    check = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if labels_r[i, j-labels_l[i, j]] == labels_l[i, j]:
                check[i, j] = 1
    
    for i in range(h):
        for j in range(w):
            if check[i, j] == 0:
                labels_l[i, j] = max_disp
                for k in range(j-1, -1, -1):
                    if check[i, k] != 0:
                        labels_l[i, j] = min(labels_l[i, j], labels_l[i, k])
                        break
                for k in range(j+1, w):
                    if check[i, k] != 0:
                        labels_l[i, j] = min(labels_l[i, j], labels_l[i, k])
                        break

    xip.weightedMedianFilter(joint=Il.astype(np.uint8), src=labels_l.astype(np.float32), dst=labels, r=11, sigma=22.5)
    
    print('Disparity Refinement takes %.4f sec'%(time.time()-t0))
    
    return labels.astype(np.uint8)
    