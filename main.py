import requests
import csv
import pandas as pd
import re
import os

from google.cloud import storage
from pytubefix import YouTube
from youtube_search import YoutubeSearch
from bs4 import BeautifulSoup

import argparse

"""
    TYPE 
    - local: save data on local
    - gcs: save data on Google Cloud Storage
"""

argparser = argparse.ArgumentParser()
argparser.add_argument('--type', help='Type of storage', default='local')
argparser.add_argument('--bucket', help='Bucket name', default='lagu-daerah')

args = argparser.parse_args()

BASE_URL = 'https://www.zonareferensi.com/lagu-daerah-indonesia/'
BUCKET_NAME = args.bucket
TYPE = args.type

def main():
    dir_check('datasets/songs')
    dir_check('data')
    
    data = None
    if os.path.exists('data/lagu_daerah.csv'):
        data = pd.read_csv('data/lagu_daerah.csv')
    else:
        data = get_song_list()
        
    dl_res = []
    for index, row in data[:5].iterrows():
        print(f"Searching for {row['Nama Lagu']}...")
        keyword = f"{row['Nama Lagu']} {row['Asal Daerah']}"
        searched_songs = search_youtube(keyword)
    

    for song in searched_songs:
        if float(song['duration']) <= 5:
            path = download_video(song['url'])
            
            dl_res.append({
                'title': song['title'],
                'region': row['Asal Daerah'],
                'keyword': f"{row['Nama Lagu']},{row['Asal Daerah']}",
                'duration': float(song['duration']),
                'url': song['url'],
                'path': path
            })
        else:
            print(f"Duration of {song['title']} is too long")
            
    df = pd.DataFrame(dl_res)
    df.to_csv('data/results.csv', index=False)
        
    
    
def dir_check(path):
    if TYPE == 'gcs':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(path)
        
        if not blob.exists():
            blob.upload_from_string('')
    else:
        if not os.path.exists(path):
            os.makedirs(path)

def get_song_list():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')
    
    data = []
    
    for row in rows:
        cols = row.find_all('td')
        
        if len(cols) == 1:
            cols = [ele.text.strip() for ele in cols]
            cols = [ele.replace(' ', '_') for ele in cols]
            data.append([ele for ele in cols if ele])
            continue
        
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
        
    if TYPE == 'gcs':
        storage_client = storage.Client()
        bucket = storage_client.bucket('lagu-daerah')
        blob = bucket.blob('lagu_daerah.csv')
        blob.upload_from_string(data)
    else:
        with open('data/lagu_daerah.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
    return data

def search_youtube(query):
    search_results = YoutubeSearch(query, max_results=5).to_dict()
    
    for i in range(len(search_results)):
        search_results[i]['url'] = 'https://www.youtube.com' + search_results[i]['url_suffix']
    
    return search_results

def yt_title_clean(title):
    text = title.lower()
    text = text.replace(' ', '_')
    text = re.sub(r'[^a-z0-9_]', '', text)
    text = re.sub(r'_{2,}', '_', text)
    
    return text
    
def download_video(video_id):
    try:
        yt = YouTube(video_id, 'WEB_EMBED')
        print(f'Downloading {yt.title}...')
        
        title = yt_title_clean(yt.title)
        song = yt.streams.get_audio_only()
        
        song.download(mp3=True, output_path='datasets/songs', filename=title)
        
        return f'datasets/songs/{title}.mp3'
    except Exception as e:
        print(f'Error: {e}')
        return None
    
    
main()
