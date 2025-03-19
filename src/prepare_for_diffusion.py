import pandas as pd
import os
import shutil
import concurrent.futures

csv_path = "/media/kursat/TOSHIBA EXT44/projects/satellite/YENI/visual-language-model/mixture_of_experts/skyscript_dataset/SkyScript_train_unfiltered_5M.csv"
images_base_dir = "/media/kursat/TOSHIBA EXT44/projects/satellite/YENI/visual-language-model/mixture_of_experts/skyscript_dataset"
train_data_dir = "/media/kursat/TOSHIBA EXT44/projects/satellite/YENI/visual-language-model/mixture_of_experts/skyscript_dataset/data_for_diffusion"

df = pd.read_csv(csv_path)
df_filtered = df[df["filepath"].str.contains("images2/")].reset_index(drop=True)

def process_row(row):
    try:
        relative_img_path = row["filepath"]  # Örneğin: 'images2/n8454739999_US_21.jpg'
        old_img_path = os.path.join(images_base_dir, relative_img_path)
        caption = row["title_multi_objects"]
        print(f"Image Path: {old_img_path}", flush=True)
        
        if not os.path.exists(old_img_path):
            print(f"Dosya bulunamadı, atlanıyor: {old_img_path}", flush=True)
            return

        original_img_name = os.path.basename(old_img_path)
        new_img_path = os.path.join(train_data_dir, original_img_name)

        # .txt dosyasının adı
        txt_filename = os.path.splitext(original_img_name)[0] + ".txt"
        new_txt_path = os.path.join(train_data_dir, txt_filename)

        # Eğer resim zaten varsa, .txt dosyasını kontrol et
        if os.path.exists(new_img_path):
            if not os.path.exists(new_txt_path):
                with open(new_txt_path, "w", encoding="utf-8") as f:
                    f.write(str(caption))
                print("Caption dosyası oluşturuldu:", new_txt_path, flush=True)
            else:
                print(f"Resim ve .txt dosyası zaten mevcut, atlanıyor: {new_img_path}", flush=True)
            return

        # Resmi kopyala
        shutil.copy(old_img_path, new_img_path)

        # Caption'ı .txt dosyasına yaz
        with open(new_txt_path, "w", encoding="utf-8") as f:
            f.write(str(caption))

        print("Dosya kaydedildi:", new_img_path, flush=True)
    except Exception as e:
        print(f"Hata oluştu: {e}", flush=True)

# Paralel işlemler için ThreadPoolExecutor kullanımı:
max_workers = 8  # Sisteminize göre ayarlayabilirsiniz
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_row, df_filtered.to_dict(orient="records"))

print(f"Toplam {len(df_filtered)} satır işlenmeye çalışıldı.", flush=True)