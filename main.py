import os
import re
import subprocess
import argparse
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import librosa
from google.cloud import storage
from youtube_search import YoutubeSearch
from bs4 import BeautifulSoup
from pydub import AudioSegment
from pytubefix import YouTube

# Constants
# BASE_URL = 'https://www.zonareferensi.com/lagu-daerah-indonesia/'
BASE_URL = 'https://dianisa.com/lagu-daerah-indonesia-beserta-lirik-dan-asalnya/'
MAX_VIDEO_DURATION = 500
SEGMENT_DURATION = 30

BUCKET_NAME = os.getenv('BUCKET_NAME')
STORAGE_TYPE = os.getenv('STORAGE_TYPE')

SONG_SEARCH_MAX_RESULTS = 15
SONG_LENGTH_PER_TITLE = 5

DOWNLOAD_SONG_LIMIT = 5


def main():
    setup_directories(['datasets/songs', 'data'])
    
    data = load_song_data()
    
    download_results = download_songs(data.head(DOWNLOAD_SONG_LIMIT))
    
    results_df = pd.DataFrame(download_results)
    results_df.to_csv('data/results.csv', index=False)
    
    results_df['wav_path'] = results_df['path'].apply(convert_to_wav)
    results_df = results_df[results_df['wav_path'].notnull()]
    results_df.to_csv('data/results_wav.csv', index=False)

    segments_df = split_songs_to_segments(results_df)
    segments_df['mfcc'] = segments_df['30s_path'].map(extract_features)
    segments_df.to_csv('data/30s_segments.csv', index=False)

    print('Extracted MFCC features for all 30s segments')
    print('Continue to training...')

def setup_directories(paths):
    for path in paths:
        if STORAGE_TYPE == 'gcs':
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(path)
            if not blob.exists():
                blob.upload_from_string('')
        else:
            os.makedirs(path, exist_ok=True)

def load_song_data():
    if os.path.exists('data/lagu_daerah.csv'):
        return pd.read_csv('data/lagu_daerah.csv')
    else:
        return get_song_list()

def get_song_list():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('table', class_='has-fixed-layout')[0]

    rows = table.find_all('tr')
    data = []
        
    for i, row in enumerate(rows):
        if i == 0:
            continue
        cols = row.find_all('td')
        
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
    
    df = pd.DataFrame(data)
    df.columns = ['nama_lagu', 'asal']
    
    if STORAGE_TYPE == 'gcs':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        blob = bucket.blob('data/lagu_daerah.csv')
        blob.upload_from_string(df.to_csv(index=False))
    else:
        df.to_csv('data/lagu_daerah.csv', index=False)
    return df

def search_youtube(query):
    search_results = YoutubeSearch(query, max_results=SONG_SEARCH_MAX_RESULTS).to_dict()
    for result in search_results:
        result['url'] = 'https://www.youtube.com' + result['url_suffix']
    return search_results

def yt_title_clean(title):
    text = title.lower().replace(' ', '_')
    return re.sub(r'[^a-z0-9_]', '', re.sub(r'_{2,}', '_', text))

def download_songs(data):
    download_results = []
    for _, row in data.iterrows():
        keyword = f"Lagu Daerah {row['nama_lagu']} asal {row['asal']}"
        searched_songs = search_youtube(keyword)
        
        downloaded_count = 0
        for song in searched_songs:
            if downloaded_count >= SONG_LENGTH_PER_TITLE:
                print(f"Downloaded {downloaded_count} songs for {row['nama_lagu']}")
                break
            try:
                duration = parse_duration(song['duration'])
                if duration < MAX_VIDEO_DURATION:
                    path = download_video(song['url'])

                    download_results.append({
                        'title': song['title'],
                        'nama_lagu': row['nama_lagu'],
                        'region': row['asal'],
                        'keyword': f"{row['asal']},{row['asal']}",
                        'duration': duration,
                        'url': song['url'],
                        'path': path
                    })
                    
                    downloaded_count += 1 
                else:
                    print(f"Duration of {song['title']} is too long")
            except ValueError as e:
                print(f"Error parsing duration for {song['title']}: {e}")
    return download_results

def parse_duration(duration_str):
    duration_str = duration_str.replace('.', ':')
    if duration_str.count(':') == 2:
        duration = datetime.strptime(duration_str, '%H:%M:%S')
    else:
        duration = datetime.strptime(duration_str, '%M:%S')
    return duration.hour * 3600 + duration.minute * 60 + duration.second

def download_video(video_id):
    try:
        yt = YouTube(video_id, 'IOS')
        print(f'Downloading {yt.title}...')
        
        title = yt_title_clean(yt.title)
        
        file_path = f'datasets/songs/{title}.mp3'
        if os.path.exists(file_path):
            print(f'{title} already exists, skipping...')
            return file_path
        
        song = yt.streams.get_audio_only()
        
        song.download(mp3=True, output_path='datasets/songs', filename=title)
        return file_path
    except Exception as e:
        print(f'Error: {e}')
        return None

def convert_to_wav(path):
    if not path:
        print(f'File is not found: {path}')
        return None
    wav_path = path.replace('songs', 'wav_songs').replace('.mp3', '.wav')
    if os.path.exists(wav_path):
        return wav_path
    os.makedirs('datasets/wav_songs', exist_ok=True)
    print(f'Converting {path} to {wav_path}')
    subprocess.run(['ffmpeg', '-i', path, wav_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path

def split_songs_to_segments(df, output_base_folder='datasets/30s_segments'):
    split_result = []
    
    for index, row in df.iterrows():
        wav_path = row['wav_path']
        nama_lagu = row['nama_lagu']
        
        setup_directories([f'{output_base_folder}/{yt_title_clean(nama_lagu)}'])
        
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio) / 1000
        num_segments = int(total_duration // SEGMENT_DURATION)
        
        for i in range(num_segments):
            start_time = i * SEGMENT_DURATION * 1000
            end_time = (i + 1) * SEGMENT_DURATION * 1000
            segment = audio[start_time:end_time]
            
            segment_file = f'{output_base_folder}/{yt_title_clean(nama_lagu)}/segment_{index}_{i}.wav'
            segment.export(segment_file, format='wav')
            
            split_result.append({
                'title': nama_lagu,
                '30s_path': segment_file
            })
            
        print(f"Saved {num_segments} segments for {nama_lagu}")
            
    return pd.DataFrame(split_result)

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None 
     
    return mfccs_processed

def extract_mfcc_features(wav_path, n_mfcc=13):
    y, sr = librosa.load(wav_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

if __name__ == '__main__':
    main()