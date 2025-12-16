# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Fungsi untuk load data dari CSV
    
    Parameter:
        file_path (str): Path ke file CSV
    
    Return:
        df (DataFrame): Data yang sudah di-load
    """
    print("="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil di-load dari: {file_path}")
        print(f"   Total baris: {len(df)}")
        print(f"   Total kolom: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return None


def remove_duplicates(df):
    """
    Fungsi untuk menghapus data duplikat
    
    Parameter:
        df (DataFrame): Data input
    
    Return:
        df_clean (DataFrame): Data tanpa duplikat
    """
    print("\n" + "="*70)
    print("STEP 2: MENGHAPUS DUPLIKASI")
    print("="*70)
    
    # Hitung duplikasi
    duplicates = df.duplicated().sum()
    print(f"Duplikasi ditemukan: {duplicates} baris")
    
    # Hapus duplikasi
    df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
    
    print(f"   Duplikasi berhasil dihapus")
    print(f"   Data sebelum: {len(df)} baris")
    print(f"   Data sesudah: {len(df_clean)} baris")
    
    return df_clean


def impute_engine_size(df):
    """
    Fungsi untuk mengisi engineSize = 0 dengan median per model
    
    Parameter:
        df (DataFrame): Data input
    
    Return:
        df (DataFrame): Data dengan engineSize yang sudah diimputasi
    """
    print("\n" + "="*70)
    print("STEP 3: IMPUTASI ENGINE SIZE = 0")
    print("="*70)
    
    # Cek berapa banyak engineSize = 0
    zero_count = (df['engineSize'] == 0).sum()
    print(f"EngineSize = 0 ditemukan: {zero_count} baris")
    
    if zero_count > 0:
        # Imputasi dengan median per model
        for model in df[df['engineSize'] == 0]['model'].unique():
            # Hitung median engineSize untuk model ini (exclude yang 0)
            median_engine = df[(df['model'] == model) & 
                              (df['engineSize'] > 0)]['engineSize'].median()
            
            # Isi yang engineSize = 0 dengan median
            df.loc[(df['model'] == model) & 
                   (df['engineSize'] == 0), 'engineSize'] = median_engine
            
            print(f"   {model}: diisi dengan median = {median_engine}")
        
        print(f" Imputasi selesai")
    else:
        print(" Tidak ada engineSize = 0, skip imputasi")
    
    return df


def remove_outliers(df, columns):
    """
    Fungsi untuk menghapus outliers menggunakan IQR method
    
    Parameter:
        df (DataFrame): Data input
        columns (list): List kolom yang akan dicek outliers
    
    Return:
        df_clean (DataFrame): Data tanpa outliers
    """
    print("\n" + "="*70)
    print("STEP 4: MENGHAPUS OUTLIERS (IQR METHOD)")
    print("="*70)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Hapus outliers untuk setiap kolom
    for col in columns:
        # Hitung Q1, Q3, dan IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Tentukan batas bawah dan atas
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Hitung outliers sebelum dihapus
        outliers_before = len(df_clean[(df_clean[col] < lower_bound) | 
                                       (df_clean[col] > upper_bound)])
        
        # Hapus outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                           (df_clean[col] <= upper_bound)]
        
        print(f"   {col}: {outliers_before} outliers dihapus")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    
    print(f"\n Outliers berhasil dihapus")
    print(f"   Data sebelum: {initial_rows} baris")
    print(f"   Data sesudah: {final_rows} baris")
    print(f"   Total dihapus: {removed_rows} baris ({removed_rows/initial_rows*100:.2f}%)")
    
    return df_clean


def encode_categorical(df, columns):
    """
    Fungsi untuk encoding kolom kategorikal menggunakan Label Encoding
    
    Parameter:
        df (DataFrame): Data input
        columns (list): List kolom kategorikal yang akan di-encode
    
    Return:
        df (DataFrame): Data dengan kolom yang sudah di-encode
        encoders (dict): Dictionary berisi encoder untuk setiap kolom
    """
    print("\n" + "="*70)
    print("STEP 5: ENCODING DATA KATEGORIKAL")
    print("="*70)
    
    encoders = {}
    
    for col in columns:
        # Buat encoder
        le = LabelEncoder()
        
        # Fit dan transform
        df[col + '_encoded'] = le.fit_transform(df[col])
        
        # Simpan encoder
        encoders[col] = le
        
        print(f"   {col} berhasil di-encode")
        print(f"   Unique values: {list(le.classes_)}")
    
    return df, encoders


def scale_features(df, numerical_columns):
    """
    Fungsi untuk scaling fitur numerik menggunakan StandardScaler
    
    Parameter:
        df (DataFrame): Data input
        numerical_columns (list): List kolom numerik yang akan di-scale
    
    Return:
        df (DataFrame): Data dengan fitur yang sudah di-scale
        scaler (StandardScaler): Scaler object
    """
    print("\n" + "="*70)
    print("STEP 6: FEATURE SCALING (STANDARDIZATION)")
    print("="*70)
    
    # Buat scaler
    scaler = StandardScaler()
    
    # Scale features
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    print(f" Feature scaling selesai untuk {len(numerical_columns)} kolom:")
    for col in numerical_columns:
        print(f"   • {col}")
    
    return df, scaler


def save_data(df, output_path):
    """
    Fungsi untuk menyimpan data hasil preprocessing
    
    Parameter:
        df (DataFrame): Data yang akan disimpan
        output_path (str): Path untuk menyimpan file
    """
    print("\n" + "="*70)
    print("STEP 7: MENYIMPAN DATA BERSIH")
    print("="*70)
    
    # Buat folder jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Simpan data
    df.to_csv(output_path, index=False)
    
    print(f" Data berhasil disimpan ke: {output_path}")
    print(f"   Total baris: {len(df)}")
    print(f"   Total kolom: {df.shape[1]}")


def preprocess_pipeline(input_path, output_path):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing
    
    Parameter:
        input_path (str): Path ke file data mentah
        output_path (str): Path untuk menyimpan data bersih
    """
    print("\n" + "="*70)
    print("AUTOMATE PREPROCESSING - TOYOTA CAR PRICE")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Step 1: Load data
    df = load_data(input_path)
    if df is None:
        return
    
    # Step 2: Hapus duplikasi
    df = remove_duplicates(df)
    
    # Step 3: Imputasi engineSize = 0
    df = impute_engine_size(df)
    
    # Step 4: Hapus outliers
    outlier_columns = ['price', 'mileage', 'mpg', 'tax']
    df = remove_outliers(df, outlier_columns)
    
    # Step 5: Encoding kategorikal
    categorical_columns = ['model', 'transmission', 'fuelType']
    df, encoders = encode_categorical(df, categorical_columns)
    
    # Step 6: Feature scaling
    numerical_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    df, scaler = scale_features(df, numerical_columns)
    
    # Step 7: Simpan data
    save_data(df, output_path)
    
    print("\n" + "="*70)
    print("PREPROCESSING SELESAI!")
    print("="*70)
    print("\nRingkasan:")
    print(f"• Data akhir: {len(df)} baris × {df.shape[1]} kolom")
    print(f"• Kolom numerik (scaled): {len(numerical_columns)}")
    print(f"• Kolom kategorikal (encoded): {len(categorical_columns)}")
    print(f"\nData siap untuk modeling! 🚀")


# ================================================================================
# MAIN PROGRAM
# ================================================================================

if __name__ == "__main__":
    """
    Program utama yang akan dijalankan ketika file ini dieksekusi
    
    Cara menjalankan:
    1. Pastikan struktur folder sudah benar
    2. Buka terminal/command prompt
    3. Jalankan: python automate_Nama-siswa.py
    """
    
    # Path ke data mentah dan data bersih
    # Sesuaikan dengan struktur folder Anda
    INPUT_PATH = '../toyota_raw/toyota.csv'  # Relative path dari folder preprocessing/
    OUTPUT_PATH = 'toyota_clean/toyota_clean.csv'  # Output di preprocessing/toyota_clean/
    
    # Jalankan pipeline preprocessing
    preprocess_pipeline(INPUT_PATH, OUTPUT_PATH)
