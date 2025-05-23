import csv
import json
from pathlib import Path

# Fungsi untuk mengonversi CSV ke JSON
def csv_to_json(csv_filename, json_filename):
    try:
        # Buat dictionary kosong
        slang_dict = {}

        # Baca file CSV
        print(f"Membaca file CSV dari: {csv_filename}")
        with open(csv_filename, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) == 2:
                    slang, formal = row
                    slang_dict[slang] = formal

        # Tulis dictionary ke file JSON
        print(f"Menulis file JSON ke: {json_filename}")
        with open(json_filename, mode='w', encoding='utf-8') as json_file:
            json.dump(slang_dict, json_file, indent=4, ensure_ascii=False)
        print("Proses konversi selesai!")

    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e.filename}")
    except PermissionError as e:
        print(f"Tidak ada izin untuk mengakses file: {e.filename}")
    except Exception as e:
        print(f"Terjadi kesalahan yang tidak terduga: {str(e)}")

# Tentukan path relatif
base_path = Path('..')
csv_filename = base_path / 'slang-word-specific.csv'
json_filename = base_path / 'slang-word-specific.json'

# Jalankan fungsi
csv_to_json(csv_filename, json_filename)