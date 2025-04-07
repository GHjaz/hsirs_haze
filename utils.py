import json
import os
import spectral.io.envi as envi

def find_deepest_directory(directory):
    """
    Функция, которая спускается в директории до самого нижнего уровня и возвращает путь.
    
    :param directory: Начальная директория для поиска.
    :return: Путь до самой глубокой директории.
    """
    deepest_path = directory
    
    # Рекурсивно обходим все поддиректории
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if not dirs:  # Если нет поддиректорий, это самая глубокая директория
            deepest_path = root
            return deepest_path
    
    return deepest_path
    
def load_labels(json_path):
    """Загружает метки из JSON-файла."""
    with open(json_path, 'r') as f:
        labels = json.load(f)
    return labels



def find_matching_files(file_list, pattern1, pattern2 = "sc01_ort"):
    """
    Функция для поиска файлов в списке, которые содержат в названии два заданных шаблона.

    :param file_list: Список имён файлов.
    :param pattern1: Первый шаблон для поиска (например, 'filename').
    :param pattern2: Второй шаблон для поиска (например, 'sc01_ort').
    :return: Список файлов, которые соответствуют обоим шаблонам.
    """
    matching_files = [file for file in file_list if pattern1 in file and pattern2 in file]
    return matching_files

def add_image_metadata(json_file, default_width, default_height):
    """
    Функция добавляет в JSON-файл ширину и высоту изображения, если их нет.

    :param json_file: Путь к JSON-файлу.
    :param default_width: Ширина изображения по умолчанию.
    :param default_height: Высота изображения по умолчанию.
    """
    # Открываем JSON-файл
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Проверяем, что data является словарём
    if not isinstance(data, dict):
        raise ValueError("JSON-файл должен содержать словарь.")

    # Добавляем ширину и высоту, если их нет
    if 'width' not in data:
        data['width'] = default_width
    if 'height' not in data:
        data['height'] = default_height

    # Сохраняем обновлённый JSON-файл
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    for i in range(7):
        json_path = f"Transform/{i+1}/labels.json"
        labels = load_labels(json_path)
        
        files = labels["files"]
        classes = labels["class"]

        
        for file, classe in zip(files, classes):
            find_way  = find_deepest_directory(file)
            files_lst = os.listdir(find_way)
            finded_files = sorted(find_matching_files(files_lst, file))
            
            print(f"{sorted(finded_files)} - {classe}")
            img_path = os.path.join(find_way, finded_files[0])
            hdr_path = os.path.join(find_way, finded_files[1])
            
            if classe == 'hazed':
                hsi_image = envi.open(hdr_path, img_path).load()
                h, w, c = hsi_image.shape
                add_image_metadata(json_path, w, h)