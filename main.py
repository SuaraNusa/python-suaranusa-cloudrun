import os
import re
import csv
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
BASE_URL = 'https://www.zonareferensi.com/lagu-daerah-indonesia/'
MAX_VIDEO_DURATION = 300
SEGMENT_DURATION = 30

BUCKET_NAME = os.getenv('BUCKET_NAME', BUCKET_NAME)
STORAGE_TYPE = os.getenv('STORAGE_TYPE', STORAGE_TYPE)


def main():
    setup_directories(['datasets/songs', 'data'])
    
    data = load_song_data()
    download_results = download_songs(data.sample(n=5))
    
    results_df = pd.DataFrame(download_results)
    results_df.to_csv('data/results.csv', index=False)
    
    results_df['wav_path'] = results_df['path'].apply(convert_to_wav)
    results_df.to_csv('data/results_wav.csv', index=False)
    
    results_df['sample_rate'], results_df['duration'] = zip(*results_df['wav_path'].map(get_duration))
    
    segments_df = split_songs_to_segments(results_df)
    segments_df['mfcc'] = segments_df['30s_path'].map(extract_mfcc_features)
    segments_df.to_csv('data/30s_segments.csv', index=False)

    print('Extracted MFCC features for all 30s segments')
    print('Continue to training...')

def setup_directories(paths):
    for path in paths:
        if TYPE == 'gcs':
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
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')
    
    data = [[ele.text.strip().replace(' ', '_') for ele in row.find_all('td')] for row in rows]
    
    if TYPE == 'gcs':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob('lagu_daerah.csv')
        blob.upload_from_string(data)
    else:
        with open('data/lagu_daerah.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
    return pd.DataFrame(data)

def search_youtube(query):
    search_results = YoutubeSearch(query, max_results=5).to_dict()
    for result in search_results:
        result['url'] = 'https://www.youtube.com' + result['url_suffix']
    return search_results

def yt_title_clean(title):
    text = title.lower().replace(' ', '_')
    return re.sub(r'[^a-z0-9_]', '', re.sub(r'_{2,}', '_', text))

def download_songs(data):
    download_results = []
    for _, row in data.iterrows():
        keyword = f"{row['Nama Lagu']} {row['Asal Daerah']}"
        searched_songs = search_youtube(keyword)
        for song in searched_songs:
            try:
                duration = parse_duration(song['duration'])
                if duration < MAX_VIDEO_DURATION:
                    path = download_video(song['url'])
                    if path:
                        download_results.append({
                            'title': song['title'],
                            'region': row['Asal Daerah'],
                            'keyword': f"{row['Nama Lagu']},{row['Asal Daerah']}",
                            'duration': duration,
                            'url': song['url'],
                            'path': path
                        })
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
        song = yt.streams.get_audio_only()
        file_path = f'datasets/songs/{title}.mp3'
        if os.path.exists(file_path):
            print(f'{title} already exists, skipping...')
            return file_path
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

def get_duration(file_path):
    if not file_path:
        print('Skipping file path is None')
        return None, None
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f'Sample rate: {sr}, duration: {duration}')
    return sr, duration

def split_songs_to_segments(df, output_base_folder='datasets/30s_segments'):
    split_result = []
    for _, row in df.iterrows():
        wav_path = row['wav_path']
        title = row['title']
        region = row['region']
        keyword = row['keyword']
        norm_title = yt_title_clean(title)
        norm_region = region.lower().replace(' ', '_')
        output_dir = os.path.join(output_base_folder, norm_region, norm_title)
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio) / 1000
        num_segments = int(total_duration // SEGMENT_DURATION)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_segments):
            start_time = i * SEGMENT_DURATION * 1000
            end_time = (i + 1) * SEGMENT_DURATION * 1000
            segment = audio[start_time:end_time]
            segment_file = os.path.join(output_dir, f"{norm_title}_segment{i + 1}.wav")
            segment.export(segment_file, format="wav")
            split_result.append({
                'title': title,
                'region': region,
                'keyword': keyword,
                '30s_path': segment_file
            })
        print(f"Saved {num_segments} segments in {norm_region} for {title} at: {output_dir}")
    return pd.DataFrame(split_result)

def extract_mfcc_features(wav_path, n_mfcc=13):
    y, sr = librosa.load(wav_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

if __name__ == '__main__':
    main()