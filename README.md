# metrics_run.py

Программа для вычисления метрик качества изображений (PSNR, SSIM, UQI, SAM, RMSE) между чистыми (clean), задымленными (hazed) и раздымленным (dehazed) изображениями.

Поддерживает как трехканальные (RGB), так и гиперспектральные (HSI) изображения.

## Структура директорий

Real/

├── real_dataset_crops/

│ └── crops_for_inference/

│ ├── data1/

│ │ ├── image1_crop0_clean.npy

│ │ ├── image1_crop0_hazed.npy

│ │ ├── image1_crop1_clean.npy

│ │ └── image1_crop1_hazed.npy

│ ├── data2/

│ │ └── ... (аналогичная структура)

│ └── ... (data3-data7)

└── results/

│ ├── data1/

│ │ ├── dehazed_crop0.npy

│ │ └── dehazed_crop1.npy

│ ├── data2/

│ │ └── ... (аналогичная структура)

│ └── ... (data3-data7)

### Требования к файлам:

1. **Исходные изображения** (в `real_dataset_crops/crops_for_inference/dataX/`):
   
   - Clean изображения: `*_crop{num}_clean.npy`
     
   - Hazed изображения: `*_crop{num}_hazed.npy`
     
   - Формат: NumPy array (.npy)
     
   - RGB: форма (H,W,3), значения [0-255]
     
   - HSI: форма (H,W,число каналов), значения [0-4096]

2. **Результаты раздымливания** (в `results/dataX/`):
   
   - Dehazed изображения: `dehazed_crop{num}.npy`
     
   - Должны соответствовать по размеру и типу исходным

## Запуск

**Запустите основной скрипт:**

*python metrics_run.py*

Результаты будут сохранены в файл metrics_results.txt

Программа генерирует отчет с метриками для каждого набора изображений.



=== Real Data Analysis for data1 ===

Crop 0 (RGB images):

Found 1 clean and 1 hazed image(s)

Metrics between hazed and clean:

PSNR: 25.1234

SSIM: 0.8765

...

=== Dehazing Results Analysis for data2 ===

Crop 0 (HSI images, found 2 clean image(s)):

Best PSNR: 29.4567

Best SSIM: 0.9234

...
