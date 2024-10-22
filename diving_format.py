import os
import pandas as pd

# Funzione per rinominare i frame nelle cartelle
def rename_frames_in_folders(base_path):
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            frames = sorted(os.listdir(folder_path))
            for idx, frame in enumerate(frames):
                old_frame_path = os.path.join(folder_path, frame)
                new_frame_name = f"{idx:05d}.jpg"
                new_frame_path = os.path.join(folder_path, new_frame_name)
                os.rename(old_frame_path, new_frame_path)
            print(f"Rinominati i frame nella cartella: {folder_name}")

# Funzione per convertire file Excel in file txt con il formato richiesto
def convert_excel_to_txt(excel_path, output_txt_path):
    # Caricamento del file Excel
    df = pd.read_excel(excel_path)
    
    with open(output_txt_path, 'w') as txt_file:
        for _, row in df.iterrows():
            folder_name = row["video"]
            label = row["label"]
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                num_frames = len(os.listdir(folder_path))
                txt_file.write(f"{folder_name} {num_frames} {label}\n")
    print(f"Creato file: {output_txt_path}")

if __name__ == "__main__":
    base_path = "/data/users/edbianchi/FRFS/FRFS/Images"  # Percorso alla cartella principale del dataset

    # Step 1: Rinominare i frame in ogni cartella
    rename_frames_in_folders(base_path)

    # Step 2: Creare i file txt equivalenti per Train e Test
    train_excel_path = "/data/users/edbianchi/FRFS/FRFS/Train.xls"
    test_excel_path = "/data/users/edbianchi/FRFS/FRFS/Test.xls"

    # specificare path di output
    convert_excel_to_txt(train_excel_path, "train_videofolder.txt")
    convert_excel_to_txt(test_excel_path, "val_videofolder.txt")
