import pandas as pd

# Load dataset
df = pd.read_csv('cyberbullying.csv')

# df.info()
# print(df.head())
# print(df.shape)

# Filter baris dengan jumlah kata di kolom 'comment' lebih dari 10
# df_filtered = df[df['comment'].apply(lambda x: len(str(x).split()) > 10)]
# print(f'Jumlah data setelah filter: {len(df_filtered)}')
# print(df_filtered.shape)

# ============================
# Pisahkan data berdasarkan kelas
df_positive = df[df['sentiment'] == -1] # Positif
df_negative = df[df['sentiment'] == 1]  # Negatif

# Hitung jumlah masing-masing kelas
count_pos = len(df_positive)
count_neg = len(df_negative)

print(f'Jumlah kelas positif: {count_pos}')
print(f'Jumlah kelas negatif: {count_neg}')

# Tentukan jumlah minimal
# min_count = min(count_pos, count_neg)

# Undersampling otomatis pada kelas mayoritas
# if count_pos > count_neg:
#     df_positive_undersampled = df_positive.sample(n=min_count, random_state=42)
#     df_cleaned = pd.concat([df_positive_undersampled, df_negative])
# elif count_neg > count_pos:
#     df_negative_undersampled = df_negative.sample(n=min_count, random_state=42)
#     df_cleaned = pd.concat([df_positive, df_negative_undersampled])
# else:
#     df_cleaned = pd.concat([df_positive, df_negative])

# Acak urutan data dan reset index
# df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan hasilnya
# df_cleaned.to_csv('cyberbullying_cleaned.csv', index=False)