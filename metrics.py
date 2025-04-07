import math
import numpy as np
from scipy.ndimage import convolve, uniform_filter

def gaussian(window_size, sigma):
    gauss = np.array([math.exp(-(x - window_size // 2)**2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size):
    _1D_window = gaussian(window_size, 1.5).reshape(-1, 1)
    _2D_window = np.dot(_1D_window, _1D_window.T)
    return _2D_window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = np.array([convolve(img1[:, :, c], window, mode='constant', cval=0.0) for c in range(channel)]).transpose(1, 2, 0)
    mu2 = np.array([convolve(img2[:, :, c], window, mode='constant', cval=0.0) for c in range(channel)]).transpose(1, 2, 0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.array([convolve(img1[:, :, c]**2, window, mode='constant', cval=0.0) for c in range(channel)]).transpose(1, 2, 0) - mu1_sq
    sigma2_sq = np.array([convolve(img2[:, :, c]**2, window, mode='constant', cval=0.0) for c in range(channel)]).transpose(1, 2, 0) - mu2_sq
    sigma12 = np.array([convolve((img1[:, :, c] * img2[:, :, c]), window, mode='constant', cval=0.0) for c in range(channel)]).transpose(1, 2, 0) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)

    ssim_map = np.clip(ssim_map, 0., 1.)
    return 1 - ssim_map

def MSE(input, target):
    mse_map = (input - target) ** 2
    return mse_map

def PSNR(input, target):
    mse_map = MSE(input, target)
    max_pixel = np.max(input, axis=(0,1))
    return 10 * np.log10(max_pixel**2 / (mse_map + 1e-8))

def RMSE(input, target):
    mse_map = MSE(input, target)
    return np.sqrt(mse_map)

def SSIM(input, target, window_size=11):
    H, W, C = input.shape
    window = create_window(window_size)
    return _ssim(input, target, window, window_size, C)


def UQI(x, y, window_size=8):
    window = create_window(window_size)  # Гауссово окно
    
    # Вычисление средних значений с помощью свертки с гауссовым окном
    meanX = np.array([convolve(x[:, :, c], window, mode='constant', cval=0.0) for c in range(x.shape[2])]).transpose(1, 2, 0)
    meanY = np.array([convolve(y[:, :, c], window, mode='constant', cval=0.0) for c in range(y.shape[2])]).transpose(1, 2, 0)

    # Вычисление дисперсий
    varX = np.array([convolve(x[:, :, c]**2, window, mode='constant', cval=0.0) for c in range(x.shape[2])]).transpose(1, 2, 0) - meanX**2
    varY = np.array([convolve(y[:, :, c]**2, window, mode='constant', cval=0.0) for c in range(y.shape[2])]).transpose(1, 2, 0) - meanY**2

    # Вычисление ковариации
    covXY = np.array([convolve(x[:, :, c] * y[:, :, c], window, mode='constant', cval=0.0) for c in range(x.shape[2])]).transpose(1, 2, 0) - meanX * meanY

    # Вычисление UQI карты
    UQI_map = 4 * meanX * meanY * covXY / ((meanX**2 + meanY**2) * (varX + varY) + 1e-12)
    
    return UQI_map  # Возвращаем среднее значение и карту

def SAM(gt: np.ndarray, pred: np.ndarray) -> tuple:
    if gt.shape != pred.shape:
        raise ValueError("Ground truth and predicted images must have the same shape.")
    
    dot_product = np.sum(gt * pred, axis=2)
    norm_gt = np.linalg.norm(gt, axis=2)
    norm_pred = np.linalg.norm(pred, axis=2)
    
    cos_theta = np.clip(dot_product / (norm_gt * norm_pred + 1e-8), -1.0, 1.0)
    sam_map = np.arccos(cos_theta)
    
    return sam_map / np.pi

def stress_metric(x:np.ndarray, y:np.ndarray):
    x_1 = np.sum(x**2)
    y_2 = np.sum(y**2)
    x_y = np.sum(x * y)
    stress_map = np.sqrt(1 - (x_y**2) / (x_1 * y_2))
    return stress_map