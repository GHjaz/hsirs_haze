import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def create_complete_metrics_csv(input_folders: List[str], output_file: str) -> None:
    """
    Создает CSV-файл с гарантированным наличием всех метрик:
    Folder,Crop,Comparison,PSNR,RMSE,SSIM
    
    Даже если некоторые метрики отсутствуют в исходных данных, они будут
    добавлены со значением None.
    """
    # Определяем полный набор ожидаемых метрик
    REQUIRED_METRICS = ['PSNR', 'SSIM',  "SAM", "UQI", 'RMSE']
    
    all_data = []
    
    for folder_path in map(Path, input_folders):
        if not folder_path.exists():
            print(f"Папка не найдена: {folder_path}")
            continue
            
        # Чтение файлов в порядке кропов
        for json_file in sorted(folder_path.glob("crop_*_metrics.json"), 
                             key=lambda x: int(x.stem.split("_")[1])):
            crop_num = int(json_file.stem.split("_")[1])
            
            with open(json_file, 'r') as f:
                try:
                    # Загружаем данные и преобразуем в нужный формат
                    metrics_data = json.load(f)
                    
                    # Создаем словарь для хранения всех метрик текущего сравнения
                    comparisons = {}
                    
                    # Обрабатываем структуру JSON (адаптируйте под ваш реальный формат)
                    if isinstance(metrics_data, dict):
                        for metric_name, comp_values in metrics_data.items():
                            if isinstance(comp_values, dict):
                                for comp_name, value in comp_values.items():
                                    if comp_name not in comparisons:
                                        comparisons[comp_name] = {
                                            'Folder': folder_path,
                                            'Crop': crop_num,
                                            'Comparison': comp_name
                                        }
                                    comparisons[comp_name][metric_name] = round(float(value), 4)
                    
                    # Добавляем все сравнения с гарантированным набором метрик
                    for comp in comparisons.values():
                        # Добавляем отсутствующие метрики как None
                        for metric in REQUIRED_METRICS:
                            if metric not in comp:
                                comp[metric] = None
                        all_data.append(comp)
                        
                except Exception as e:
                    print(f"Ошибка обработки {json_file}: {str(e)}")
                    continue
    
    if not all_data:
        print("Нет данных для обработки")
        return
    
    # Создаем DataFrame
    df = pd.DataFrame(all_data)
    
    # Убедимся, что все требуемые колонки присутствуют
    for metric in REQUIRED_METRICS:
        if metric not in df.columns:
            df[metric] = None
    
    # Выбираем и сортируем колонки
    columns_order = ['Folder', 'Crop', 'Comparison'] + REQUIRED_METRICS
    df = df[columns_order].sort_values(['Folder', 'Crop', 'Comparison'])
    
    # Сохраняем в CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Файл с полным набором метрик сохранен в: {output_file}")

# Пример использования
if __name__ == "__main__":
    folders = [
        "1/results/",
        "2/results/",
        "3/results/",
        "4/results/",
        "5/results/",
        "6/results/",
        "7/results/"
    ]
    
    create_complete_metrics_csv(
        input_folders=folders,
        output_file="results/complete_metrics_report_v4.csv"
    )