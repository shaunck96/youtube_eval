import pandas as pd
import scrapetube
from pytube import YouTube
import os
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import openai
import regex as re
import json
import requests
from tqdm import tqdm
from googleapiclient.discovery import build
import regex as re
from transformers import pipeline
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from faster_whisper import WhisperModel
import tiktoken
import pandas as pd
import re
from urllib import parse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from itertools import islice
from youtube_comment_downloader import *
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from itertools import islice
from youtube_comment_downloader import *
from pytube import YouTube
import cv2
import base64
import requests
from PIL import Image
from pathlib import Path
from moviepy import editor
import os
import torch
#from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
#from diffusers.utils import export_to_video
import imageio
import os
import requests
import googleapiclient.discovery
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.llms import OpenAI
import requests
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def url_transcriber(url_list, output_path):
  #for i in range(len(url_list)):
  url_mp3_mapping = {}
  for i in range(len(url_list)):
    yt = YouTube(url_list[i])
    video = yt.streams.filter(only_audio=True).first()
    destination = output_path
    out_file = video.download(output_path=destination)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    url_mp3_mapping[new_file] = url_list[i]
    os.rename(out_file, new_file)
    print(yt.title + " has been successfully downloaded.")
  return url_mp3_mapping

def transcriber(mp3_details, model):
  transcriptions={}
  transcription_by_time={}

  segments, info = model.transcribe(list(mp3_details.keys())[0], beam_size=5, task='translate')
  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
  for segment in segments:
      print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
      transcription_by_time[str(segment.start)+'_'+str(segment.end)] = segment.text

  transcriptions[list(mp3_details.values())[0]] = transcription_by_time
  return(transcriptions)

def split_into_chunks(text, chunk_size=4000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def cleanup_transcript(text):
    # Make an API call to OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Act as a university graduate with a masters degree in English. I will provide you with a transcription from a video. You will provide a reformatted copy corrected to follow rules of English grammar.\nTranscript:\n{text}\Copy with corrected English:",
        temperature=0,
        top_p=1,
        # the number of tokens in a string can be estimated by multiplying the number of characters by 1.3 and rounding up
        # max_tokens=round(len(transcript_text) * 1.3)
        max_tokens=2000
    )

    print(response)
    return response["choices"][0]["text"]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def video_id_from_url(url):
  url_parsed = parse.urlparse(url)
  qsl = parse.parse_qs(url_parsed.query)
  return qsl['v'][0]

def get_video_categories(api_key, region_code='US'):
    # Build the YouTube client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Make a request to get video categories
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code
    )
    response = request.execute()

    # Process the response to extract category IDs and names
    categories = []
    for item in response['items']:
        categories.append({
            'category_id': item['id'],
            'category_name': item['snippet']['title']
        })

    return pd.DataFrame(categories)

# Building the YouTube client
def get_trending_videos(category_id, max_results):
  trending_by_category = pd.DataFrame(columns=['category','title','viewCount','publishDate','video_id','url'])
  # Fetching trending videos
  youtube = build('youtube', 'v3', developerKey=api_key)
  request = youtube.videos().list(
      part="snippet,contentDetails,statistics",
      chart="mostPopular",
      regionCode="US",  # Change to your region
      videoCategoryId=category_id,
      #maxResults=max_results
  )
  response = request.execute()

  for video in response['items']:
      video_id = video['id']
      video_url = f"https://www.youtube.com/watch?v={video_id}"
      print(f"Title: {video['snippet']['title']}")
      print(f"Views: {video['statistics']['viewCount']}")
      print(f"Published at: {video['snippet']['publishedAt']}")
      trending_by_category = pd.concat([trending_by_category,pd.DataFrame({'category':category_id,'title':[video['snippet']['title']],'viewCount':[video['statistics']['viewCount']],'publishDate':[video['snippet']['publishedAt']],'video_id':[video_id],'url':f"https://www.youtube.com/watch?v={video_id}"})])
  return trending_by_category

def thumbnail_extractor(url):
  # URL of the image
  image_url = "https://img.youtube.com/vi/Wdjh81uH6FU/0.jpg"

  # Send a GET request to the URL
  response = requests.get(image_url)

  # Check if the request was successful
  if response.status_code == 200:
      # Write the content of the response to a file
      with open("/content/thumbnails/{}.jpg".format(str(video_id_from_url(url))), "wb") as file:
          file.write(response.content)

def youtube_video_categories_extractor(api_key):
  api_key = 'AIzaSyBRvAdrZonkE--IGTqU22So6Dnt-6xfJg0'
  category_dict = {}
  # Define the API endpoint
  endpoint = "https://www.googleapis.com/youtube/v3/videoCategories"

  # Define the query parameters
  params = {
      "key": api_key,
      "part": "snippet",
      "regionCode": "US"  # You can change the region code if needed
  }

  # Make the GET request
  response = requests.get(endpoint, params=params)

  # Check if the request was successful
  if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract and print the available video categories
    for category in data.get("items", []):
        category_id = category["id"]
        category_title = category["snippet"]["title"]
        category_dict[category_id] = category_title
    return category_dict
  else:
    print(f"Request failed with status code: {response.status_code}")

def video_comments(api_key,url):
  video_id = video_id_from_url(url)
  # List to store comments
  comments_list = []

  # Creating YouTube resource object
  youtube = build('youtube', 'v3', developerKey=api_key)

  # Retrieve YouTube video results
  video_response = youtube.commentThreads().list(
      part='snippet,replies',
      videoId=video_id
  ).execute()

  # Iterate video response
  while video_response:
      for item in video_response['items']:
          # Extracting comment
          comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
          comments_list.append(comment)

          # Check if there are replies to the comment
          replycount = item['snippet']['totalReplyCount']
          if replycount > 0:
              # Iterate through all replies
              for reply in item['replies']['comments']:
                  # Extract and add reply to the list
                  reply_text = reply['snippet']['textDisplay']
                  comments_list.append(reply_text)

      # Check if there are more pages
      if 'nextPageToken' in video_response:
          video_response = youtube.commentThreads().list(
              part='snippet,replies',
              videoId=video_id,
              pageToken=video_response['nextPageToken']
          ).execute()
      else:
          break
  return(comments_list)