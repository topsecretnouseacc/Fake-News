import pandas as pd

def load_and_merge(raw_dir='data/raw'):
    # Fake ve real CSV’lerini oku, label ekle
    fake = pd.read_csv(f'{raw_dir}/fake.csv')
    fake['label'] = 1
    real = pd.read_csv(f'{raw_dir}/true.csv')
    real['label'] = 0
    
    # Birleştir ve karıştır
    df = pd.concat([fake, real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def preprocess(raw_dir='data/raw', out_path='data/processed/combined.csv'):
    df = load_and_merge(raw_dir)
    
    # Başlık + metni birleştir, eksikleri at
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df[['content', 'label']]
    
    # Kaydet
    df.to_csv(out_path, index=False)
    print(f'Önişleme tamam. {out_path} dosyası oluşturuldu. Satır sayısı: {len(df)}')

if __name__ == '__main__':
    preprocess()
