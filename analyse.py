import numpy as np
import json
import os
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
from metrics import PSNR, SSIM, UQI, SAM, RMSE

wls = {"wavelength": [365.9298, 375.594, 385.2625, 394.9355, 404.6129, 414.2946, 423.9808, 433.6713, 443.3662, 453.0655, 462.7692, 472.4773, 482.1898, 491.9066, 501.6279, 511.3535, 521.0836, 530.818, 540.5568, 550.3, 560.0477, 569.7996, 579.556, 589.3168, 599.0819, 608.8515, 618.6254, 628.4037, 638.1865, 647.9736, 657.7651, 667.561, 654.7923, 664.5994, 674.4012, 684.1979, 693.9894, 703.7756, 713.5566, 723.3325, 733.1031, 742.8685, 752.6287, 762.3837, 772.1335, 781.8781, 791.6174, 801.3516, 811.0805, 820.8043, 830.5228, 840.2361, 849.9442, 859.6471, 869.3448, 879.0372, 888.7245, 898.4066, 908.0834, 917.7551, 927.4214, 937.0827, 946.7387, 956.3895, 966.0351, 975.6755, 985.3106, 994.9406, 1004.565, 1014.185, 1023.799, 1033.408, 1043.012, 1052.611, 1062.204, 1071.793, 1081.376, 1090.954, 1100.526, 1110.094, 1119.656, 1129.213, 1138.765, 1148.311, 1157.853, 1167.389, 1176.92, 1186.446, 1195.966, 1205.482, 1214.992, 1224.497, 1233.996, 1243.491, 1252.98, 1262.464, 1252.773, 1262.746, 1272.718, 1282.691, 1292.662, 1302.634, 1312.606, 1452.182, 1462.15, 1472.118, 1482.085, 1492.052, 1502.019, 1511.986, 1521.952, 1531.918, 1541.885, 1551.85, 1561.816, 1571.781, 1581.746, 1591.711, 1601.675, 1611.64, 1621.604, 1631.568]}

class HSIMetricCalculator:
    def __init__(self, metrics=None):
        self.metrics = metrics or [PSNR, SSIM, UQI, SAM, RMSE]
        self.metric_ranges = {
            "PSNR": (10, 30),  # Типичный диапазон значений
            "SSIM": (0, 1),
            "UQI": (0, 1),
            "SAM": (0, 1),  
            "RMSE": (0, 1)  # Для нормализованных данных
        }

    @staticmethod
    def load_image(file_path, class_name, crop_num):
        data = np.load(file_path).astype(np.float32)
        data = np.clip(data / 4096.0, 0., 1.)
        return {
            "data": data,
            "class_name": class_name,
            "file_name": Path(file_path).stem,
            "crop_num": crop_num
        }
    
    def compute_metric(self, metric, img1, img2):
        metric_map = metric(img1["data"], img2["data"])
        return np.mean(metric_map), metric_map

class HSIResultsHandler:
    @staticmethod
    def save_results(results, output_dir, crop_num):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"crop_{crop_num}_metrics.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    @staticmethod
    def print_summary(results, crop_num):
        print(f"\n{'='*40}")
        print(f"Crop {crop_num} Metrics Summary")
        print(f"{'='*40}")
        for metric_name, comparisons in results.items():
            sign = '↑' if metric_name not in ['RMSE', 'SAM'] else '↓'
            print(f"\n{metric_name} ({sign}):")
            for pair, value in comparisons.items():
                print(f"  {pair}: {value:.4f}")

def main():
    calculator = HSIMetricCalculator()
    results_handler = HSIResultsHandler()
    
    for i in range(1, 8):
        config = {
            "data_dir": f"{i}/",
            "json_path": f"{i}/labels.json",
            "output_dir": f"{i}/results"
        }
        
        with open(config["json_path"]) as f:
            labels_config = json.load(f)
    
        config["num_crops"] = len(labels_config['coordinates'])
        print(f'Folder: {config["data_dir"]} in progress')
        
        for crop_num in range(1, config["num_crops"] + 1):
            images = []
            
            for basename, class_name in zip(labels_config["files"], labels_config["class"]):
                file_path = Path(config["data_dir"]) / f"{basename}_crop{crop_num}.npy"
                try:
                    img = calculator.load_image(str(file_path), class_name, crop_num)
                    images.append(img)
                except FileNotFoundError:
                    print(f"Warning: File not found {file_path}")
                    continue
            
            if not images:
                continue

            metrics_results = {metric.__name__: {} for metric in calculator.metrics}

            pairs = list(combinations(images, 2))
            for metric in calculator.metrics:
                metric_name = metric.__name__
                vmin, vmax = calculator.metric_ranges.get(metric_name, (0, 1))
                metric_maps_dir = Path(config["output_dir"]) / "metric_maps_by_channels" / metric_name
                metric_maps_dir.mkdir(parents=True, exist_ok=True)
                
                fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 5))
                fig.suptitle(f"Metric: {metric.__name__} for Crop {crop_num}")
                # Обработка случая, если axes - не массив, а одна ось
                if len(pairs) == 1:
                    axes = [axes]  

                
                for ax, (img1, img2) in zip(axes, pairs):
                    pair_name = f"{img1['class_name']}_{img1['file_name'][:7]} vs {img2['class_name']}_{img2['file_name'][:7]}"
                    value, metric_map = calculator.compute_metric(metric, img1, img2)
                    metrics_results[metric_name][pair_name] = float(value)
                    if metric_map.ndim == 3:
                        for channel in range(0, metric_map.shape[2], 5):
                            channel_map = metric_map[:, :, channel]
                            plt.figure(figsize=(10, 5))
                            plt.imshow(channel_map, cmap='inferno', vmin=vmin, vmax=vmax)
                            plt.colorbar()
                            plt.title(f"{metric_name} | {pair_name} | Wavelength/nm {wls['wavelength'][channel]}")
                            plt.axis('off')
                            plt.savefig(metric_maps_dir / f"crop_{crop_num}_{pair_name}_ch{channel}.png")
                            plt.close()
                        metric_map = np.mean(metric_map, axis=2)
                    ax.imshow(metric_map, cmap='inferno')
                    ax.set_title(pair_name, fontsize=8)
                    ax.axis('off')
                # Сохранение карты метрик на холсте
                metric_maps_dir = Path(config["output_dir"]) / "metric_maps"
                metric_maps_dir.mkdir(parents=True, exist_ok=True)
                plt.colorbar(axes[0].imshow(metric_map, cmap='inferno'), ax=axes, shrink=0.8)
                plt.savefig(metric_maps_dir / f"crop_{crop_num}_{metric.__name__}_combined.png")
                plt.close()
                    

            results_handler.save_results(metrics_results, config["output_dir"], crop_num)
            results_handler.print_summary(metrics_results, crop_num)

if __name__ == "__main__":
    main()