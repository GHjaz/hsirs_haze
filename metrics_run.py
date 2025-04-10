import numpy as np
from itertools import combinations
from pathlib import Path
from metrics import PSNR, SSIM, UQI, SAM, RMSE

class ImageMetricCalculator:
    def __init__(self, metrics=None):
        self.metrics = metrics or [PSNR, SSIM, UQI, SAM, RMSE]
        self.hsi_wavelengths = [365.9298, 375.594, 385.2625, 394.9355, 404.6129, 414.2946, 
                               423.9808, 433.6713, 443.3662, 453.0655, 462.7692, 472.4773, 
                               482.1898, 491.9066, 501.6279, 511.3535, 521.0836, 530.818, 
                               540.5568, 550.3, 560.0477, 569.7996, 579.556, 589.3168, 
                               599.0819, 608.8515, 618.6254, 628.4037, 638.1865, 647.9736, 
                               657.7651, 667.561, 654.7923, 664.5994, 674.4012, 684.1979, 
                               693.9894, 703.7756, 713.5566, 723.3325, 733.1031, 742.8685, 
                               752.6287, 762.3837, 772.1335, 781.8781, 791.6174, 801.3516, 
                               811.0805, 820.8043, 830.5228, 840.2361, 849.9442, 859.6471, 
                               869.3448, 879.0372, 888.7245, 898.4066, 908.0834, 917.7551, 
                               927.4214, 937.0827, 946.7387, 956.3895, 966.0351, 975.6755, 
                               985.3106, 994.9406, 1004.565, 1014.185, 1023.799, 1033.408, 
                               1043.012, 1052.611, 1062.204, 1071.793, 1081.376, 1090.954, 
                               1100.526, 1110.094, 1119.656, 1129.213, 1138.765, 1148.311, 
                               1157.853, 1167.389, 1176.92, 1186.446, 1195.966, 1205.482, 
                               1214.992, 1224.497, 1233.996, 1243.491, 1252.98, 1262.464, 
                               1252.773, 1262.746, 1272.718, 1282.691, 1292.662, 1302.634, 
                               1312.606, 1452.182, 1462.15, 1472.118, 1482.085, 1492.052, 
                               1502.019, 1511.986, 1521.952, 1531.918, 1541.885, 1551.85, 
                               1561.816, 1571.781, 1581.746, 1591.711, 1601.675, 1611.64, 
                               1621.604, 1631.568]

    def determine_image_type(self, img_array):
        if len(img_array.shape) != 3:
            raise ValueError(f"Invalid image dimensions: {img_array.shape}. Expected 3D array (H,W,C)")
            
        num_channels = img_array.shape[2]
        
        if num_channels == 3:
            return 'RGB'
        elif num_channels == len(self.hsi_wavelengths):
            return 'HSI'
        else:
            raise ValueError(f"Unknown image type with {num_channels} channel(s). Expected 3 (RGB) or {len(self.hsi_wavelengths)} (HSI)")

    def load_image(self, file_path):
        data = np.load(file_path).astype(np.float32)
        img_type = self.determine_image_type(data)
        
        if img_type == 'RGB':
            return np.clip(data / 255.0, 0., 1.)
        elif img_type == 'HSI':
            return np.clip(data / 4096.0, 0., 1.)

    def compute_metric(self, metric, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
        
        type1 = self.determine_image_type(img1)
        type2 = self.determine_image_type(img2)
        if type1 != type2:
            raise ValueError(f"Cannot compare different image types: {type1} vs {type2}")
        
        metric_map = metric(img1, img2)
        return np.mean(metric_map)


def analyze_real_data(data_dir, output_file):
    calculator = ImageMetricCalculator()
    
    with open(output_file, 'a') as f:
        f.write(f"\n=== Real Data Analysis for {data_dir.name} ===\n")
        
        crop_nums = set()
        for file in data_dir.glob("*_crop*.npy"):
            crop_num = int(file.stem.split('_crop')[-1].split('_')[0])
            crop_nums.add(crop_num)
        
        for crop_num in sorted(crop_nums):
            crop_files = list(data_dir.glob(f"*_crop{crop_num}_*.npy"))
            clean_imgs = sorted([f for f in crop_files if "clean" in f.stem])
            hazed_imgs = [f for f in crop_files if "hazed" in f.stem]
            
            if not clean_imgs or not hazed_imgs:
                continue
                
            try:
                hazed = calculator.load_image(hazed_imgs[0])
                cleans = [calculator.load_image(f) for f in clean_imgs]
                
                img_type = calculator.determine_image_type(hazed)
                f.write(f"\nCrop {crop_num} ({img_type} images):\n")
                f.write(f"Found {len(clean_imgs)} clean and {len(hazed_imgs)} hazed image(s)\n")
                
                if len(clean_imgs) == 1 and len(hazed_imgs) == 1:
                    f.write("Metrics between hazed and clean:\n")
                    for metric in calculator.metrics:
                        value = calculator.compute_metric(metric, hazed, cleans[0])
                        f.write(f"{metric.__name__}: {value:.4f}\n")
                
                elif len(clean_imgs) > 1 and len(hazed_imgs) == 1:
                    hazed_clean_rmses = [calculator.compute_metric(RMSE, hazed, clean) for clean in cleans]
                    clean_pairs = list(combinations(range(len(clean_imgs)), 2))
                    clean_clean_rmses = []
                    
                    f.write("\nClean image comparisons:\n")
                    for i, j in clean_pairs:
                        rmse = calculator.compute_metric(RMSE, cleans[i], cleans[j])
                        clean_clean_rmses.append(rmse)
                        f.write(f"RMSE between {clean_imgs[i].stem} and {clean_imgs[j].stem}: {rmse:.4f}\n")
                    
                    R = np.mean(hazed_clean_rmses) / np.mean(clean_clean_rmses)
                    f.write(f"\nR metric: {R:.4f}\n")
                    
                    for i, clean in enumerate(cleans, 1):
                        f.write(f"\nHazed vs {clean_imgs[i-1].stem} metrics:\n")
                        for metric in calculator.metrics:
                            value = calculator.compute_metric(metric, hazed, clean)
                            f.write(f"{metric.__name__}: {value:.4f}\n")
            
            except Exception as e:
                f.write(f"\nError processing crop {crop_num}: {str(e)}\n")
                continue


def analyze_dehazing_results(real_data_dir, dehazed_dir, output_file):
    calculator = ImageMetricCalculator()
    
    with open(output_file, 'a') as f:
        f.write(f"\n\n=== Dehazing Results Analysis for {real_data_dir.name} ===\n")
        
        crop_nums = set()
        for file in real_data_dir.glob("*_crop*_clean.npy"):
            crop_num = int(file.stem.split('_crop')[-1].split('_')[0])
            crop_nums.add(crop_num)
        
        for crop_num in sorted(crop_nums):
            clean_imgs = list(real_data_dir.glob(f"*_crop{crop_num}_clean.npy"))
            dehazed_path = dehazed_dir / f"dehazed_crop{crop_num}.npy"
            
            if not dehazed_path.exists():
                continue
                
            try:
                dehazed = calculator.load_image(dehazed_path)
                cleans = [calculator.load_image(f) for f in clean_imgs]
                
                img_type = calculator.determine_image_type(dehazed)
                f.write(f"\nCrop {crop_num} ({img_type} images, found {len(clean_imgs)} clean image(s)):\n")
                
                results = {}
                for metric in calculator.metrics:
                    metric_name = metric.__name__
                    values = [calculator.compute_metric(metric, dehazed, clean) for clean in cleans]
                    
                    if metric_name in ['PSNR', 'SSIM', 'UQI']:
                        best_value = max(values)
                    else:
                        best_value = min(values)
                    
                    results[metric_name] = best_value
                
                for metric_name, value in results.items():
                    f.write(f"Best {metric_name}: {value:.4f}\n")
            
            except Exception as e:
                f.write(f"\nError processing crop {crop_num}: {str(e)}\n")
                continue


def main():
    base_dir = Path("Real")
    output_file = "metrics_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("=== Metrics Analysis Results ===\n")
    
    for data_num in range(1, 8):
        real_data_dir = base_dir / f"real_dataset_crops/crops_for_inference/data{data_num}"
        dehazed_dir = base_dir / f"results/data{data_num}"
        
        if not real_data_dir.exists():
            print(f"Directory {real_data_dir} not found, skipping...")
            continue
            
        analyze_real_data(real_data_dir, output_file)
        
        if not dehazed_dir.exists():
            print(f"Directory {dehazed_dir} not found, skipping dehazing analysis...")
            continue
            
        analyze_dehazing_results(real_data_dir, dehazed_dir, output_file)
        
    print(f"\nAll metrics saved to {output_file}")

if __name__ == "__main__":
    main()
