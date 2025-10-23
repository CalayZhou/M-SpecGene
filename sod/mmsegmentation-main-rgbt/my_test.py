import numpy as np
import torch

def S_object(prediction, GT):
    # Compute the object similarity
    prediction_fg = prediction.clone()
    prediction_fg[~GT] = 0
    O_FG = Object(prediction_fg, GT)

    prediction_bg = 1.0 - prediction
    prediction_bg[GT] = 0
    O_BG = Object(prediction_bg, ~GT)

    u = torch.mean(GT.float())
    Q = u * O_FG + (1 - u) * O_BG

    return Q

def Object(prediction, GT):
    if prediction.numel() == 0:
        return 0

    if not prediction.dtype == torch.float64:
        prediction = prediction.double()

    if (prediction.max() > 1) or (prediction.min() < 0):
        raise ValueError('Prediction should be in the range of [0, 1]')
    if not GT.dtype == torch.bool:
        raise ValueError('GT should be of type: logical')

    if GT.sum() == 0:  # 如果GT全是0，返回0
        return 0

    x = prediction[GT].mean()
    sigma_x = prediction[GT].std()

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + 1e-10)
    return score

def S_region(prediction, GT):
    X, Y = centroid(GT)
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(GT, X, Y)
    prediction_1, prediction_2, prediction_3, prediction_4 = Divideprediction(prediction, X, Y)

    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)

    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def centroid(GT):
    rows, cols = GT.shape
    total = GT.sum()

    if total == 0:
        X = cols // 2  # 使用整数除法
        Y = rows // 2  # 使用整数除法
    else:
        i = torch.arange(1, cols + 1, dtype=torch.float64)
        j = torch.arange(1, rows + 1, dtype=torch.float64).reshape(-1, 1)
        X = int((GT.sum(dim=0) * i).sum() / total)  # 转换为整数
        Y = int((GT.sum(dim=1) * j).sum() / total)  # 转换为整数

    return X, Y


def divideGT(GT, X, Y):
    hei, wid = GT.shape
    area = wid * hei

    LT = GT[:Y, :X]
    RT = GT[:Y, X:wid]
    LB = GT[Y:hei, :X]
    RB = GT[Y:hei, X:wid]

    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei - Y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4

def Divideprediction(prediction, X, Y):
    hei, wid = prediction.shape

    LT = prediction[:Y, :X]
    RT = prediction[:Y, X:wid]
    LB = prediction[Y:hei, :X]
    RB = prediction[Y:hei, X:wid]

    return LT, RT, LB, RB

def ssim(prediction, GT):
    dGT = GT.float()
    hei, wid = prediction.shape
    N = wid * hei

    x = prediction.mean()
    y = dGT.mean()

    eps = 1e-10
    sigma_x2 = ((prediction - x) ** 2).sum() / (N - 1 + eps)
    sigma_y2 = ((dGT - y) ** 2).sum() / (N - 1 + eps)
    sigma_xy = ((prediction - x) * (dGT - y)).sum() / (N - 1 + eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

    if beta > 0:  # 只在beta > 0时计算Q
        Q = alpha / beta
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q

def StructureMeasure(prediction, GT):
    if not prediction.dtype == torch.float64:
        raise ValueError('The prediction should be double type...')
    if (prediction.max() > 1) or (prediction.min() < 0):
        raise ValueError('The prediction should be in the range of [0, 1]...')
    if not GT.dtype == torch.bool:
        raise ValueError('GT should be logical type...')

    y = GT.float().mean()

    if y == 0:
        x = prediction.mean()
        Q = 1.0 - x
    elif y == 1:
        x = prediction.mean()
        Q = x
    else:
        alpha = 0.5
        O = S_object(prediction, GT)
        R = S_region(prediction, GT)

        # 确保 O 和 R 在 0 到 1 的范围内
        O = max(0, min(O, 1))
        R = max(0, min(R, 1))

        Q = alpha * O + (1 - alpha) * R
        Q = max(Q, 0)
        Q = min(Q, 1)  # 确保最终的Q在0到1之间

    return Q



if __name__ == "__main__":
    # Create example prediction and ground truth tensors
    prediction = torch.tensor([[0, 0, 0, 0],
                               [0, 1, 1, 0],
                               [0, 1, 1, 0],
                               [0, 0, 0, 0]], dtype=torch.float64)  # Example prediction

    GT = torch.tensor([[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0]], dtype=torch.bool)  # Example ground truth

    # Calculate similarity score
    Q = StructureMeasure(prediction, GT)
    print('Similarity score:', Q.item())  # Convert tensor to Python float for printing
