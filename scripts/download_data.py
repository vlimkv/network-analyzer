import os
import requests
import zipfile

def download_file(url, save_path):
    """
    Скачивает файл по указанному URL и сохраняет его по заданному пути.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Проверка на ошибки

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Файл сохранен: {save_path}")

def extract_zip(file_path, extract_to):
    """
    Извлекает zip-архив в заданную директорию.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Файл {file_path} извлечен в директорию {extract_to}")

def main():
    # Создаем директорию data, если она не существует
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Список файлов для скачивания
    datasets = {
        'NSL-KDD.zip': 'https://cloudstor.aarnet.edu.au/plus/s/YNu2itxiK73ZGdl/download',
        'UNSW-NB15.zip': 'https://cloudstor.aarnet.edu.au/plus/s/7XOjVQBfI8kanfC/download'
    }

    for filename, url in datasets.items():
        save_path = os.path.join(data_dir, filename)
        print(f"Скачивание {filename}...")
        download_file(url, save_path)

        # Проверяем, является ли файл zip-архивом, и извлекаем его
        if filename.endswith('.zip'):
            extract_zip(save_path, data_dir)
            # Удаляем zip-файл после извлечения
            os.remove(save_path)

if __name__ == "__main__":
    main()
