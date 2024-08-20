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
from .utils import *

class YoutubeVideoAIStatExtractor:

  def __init__(self, url, youtube_api_key="AIzaSyBRvAdrZonkE--IGTqU22So6Dnt-6xfJg0", open_ai_api_key="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"):
    self.openai_api_key = open_ai_api_key
    self.api_key = youtube_api_key
    self.model_size = "small"
    self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
    self.transcriptions = {}
    self.formatted_transcriptions = []
    self.audio_file_output_path = r'/content/audio_downloads'
    self.mp3_details = url_transcriber([url],self.audio_file_output_path)
    self.transcriptions = transcriber(self.mp3_details,self.model)
    self.complete_content = []
    self.time_based_content = []
    self.clean_transcript = []
    self.complete_content_chunks = []
    self.topic_df = pd.DataFrame(columns=['video_id','topics'])
    self.sentiment_df = pd.DataFrame(columns=['url','sentiment_topics'])
    self.downloader = YoutubeCommentDownloader()
    self.trending_by_category = pd.DataFrame(columns=['url', 'theme', 'introduction', 'conclusion', 'topic_of_discussion', 'category', 'num_of_speakers', 'keywords', 'video_type', 'summary', 'tone', 'num_of_scenes'])

  def transcription_entities_extractor(self):
    response_schemas = [
        ResponseSchema(name="URL", description="Entire Youtube Link"),
        ResponseSchema(name="Overall Theme", description="Overall Theme of the Conversation"),
        ResponseSchema(name="Introduction", description="Describe how the content is introduced"),
        ResponseSchema(name="Conclusion", description="Describe how the content creator concludes the video"),
        ResponseSchema(name="Topics of Discussion", description="Different topics discussed in the transcription"),
        ResponseSchema(name="Category", description="High Level Category of Video based on content"),
        ResponseSchema(name="Number of Speakers", description="Number of Speakers in the Video"),
        ResponseSchema(name="Keywords", description="Keywords/Phrases relevant to topic used"),
        ResponseSchema(name="Type of Video", description="Video Tutorials,Ask Me Anything,Whiteboard Video,Listicle Video,Product Review,Educational Video,Challenge Video,Unboxing Video,Behind the Scenes,Explainer Video,Product Demo,Video Testimonial,Reaction Video,Webinar Teaser,Community-Based Video,Q&A Video,Video Blogs,Product Launch,Video Podcast"),
        ResponseSchema(name="Summary", description="Summary of the entire video"),
        ResponseSchema(name="Tone", description="Tone of conversation used"),
        ResponseSchema(name="Number of Scenes", description="Estimated Number of Scenes"),
        ResponseSchema(name="Related Topics", description="List of related sub topics not discussed above")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5, openai_api_key = self.openai_api_key)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("You are a helpful assistant who evaluates transcriptions of youtube videos, identifies topic of discussion and summarizes it in a concise format.\n{format_instructions}\nTranscriptions: {question}")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    topic_dict = {}
    token_check = 0
    self.clean_transcript+='\nURL: '+str(list(self.transcriptions.keys())[0])
    transcription_token_length = num_tokens_from_string(self.clean_transcript, "gpt-3.5-turbo-16k")
    if transcription_token_length<16000:
      _input = prompt.format_prompt(question=self.clean_transcript)
      output = chat_model(_input.to_messages())
      topic_dict[output_parser.parse(output.content)['URL']] = output_parser.parse(output.content)
      print(output_parser.parse(output.content))
    else:
      print("Token Limit Exceeded. Summarizing and evaluating")
      complete_content_chunks = split_into_chunks(self.clean_transcript,16000)
      summarized_transcription = []
      for chunk in complete_content_chunks:
        doc =  Document(page_content=chunk, metadata={"source": "transcription"})
        summ_chain = load_summarize_chain(chat_model, chain_type="stuff")
        transcription = summ_chain.run([doc])
        summarized_transcription.append(transcription)
      summarized_transcription = ' '.join(summarized_transcription)+'\nURL: '+str(list(self.transcriptions.keys())[0])
      summ_chain = load_summarize_chain(chat_model, chain_type="stuff")
      doc_summarized =  Document(page_content=summarized_transcription, metadata={"source": "summarized_transcription"})
      summarized_transcription_updated = summ_chain.run([doc_summarized])
      _input = prompt.format_prompt(question=summarized_transcription_updated)
      output = chat_model(_input.to_messages())
      topic_dict[output_parser.parse(output.content)['URL']] = output_parser.parse(output.content)
      print(output_parser.parse(output.content))

    self.topic_df['url'] = list(list(topic_dict.keys()))
    self.topic_df['topics'] = list(list(topic_dict.values()))
    self.sentiment_extractor()

  def sentiment_extractor(self):
    urls_and_comments = {}
    url = self.topic_df['url'].iloc[0]
    comments = self.downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    comment_list = []

    for comment in islice(comments, 30):
      comment_list.append(comment['text'])

    urls_and_comments[url] = '\n'.join(comment_list)

    pos_neg_sentiment_topics = {}

    for url, comment_chunk in urls_and_comments.items():

      sentiment_prompt_template = '''
    Evaluate the following comments and assign a sentiment label to the youtube video.
    Return the topics indicating the sentiment as well.

    Youtube Video Comments split by \n: {comment_chunk}

    Format instructions:
    Stick to the below JSON output format and do not return any additional information. Be precise and only use the context to come up with a solution.

    JSON SCHEMA:
    ""{{"positive_sentiment": "List of topics with one line description of topic indicating a positive sentiment","negative_sentiment": "List of topics with one line description of topics indicating a negative sentiment"}}""
      '''

      prompt = PromptTemplate(
          template=sentiment_prompt_template,
          input_variables=["query"]
          #,partial_variables={"format_instructions": parser.get_format_instructions()},
      )
      _input = prompt.format_prompt(comment_chunk=comment_chunk)

      chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key = "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")

      response = chat_model(_input.to_messages())

      pos_neg_sentiment_topics[url] = response

    self.sentiment_df['url'] = list(pos_neg_sentiment_topics.keys())
    self.sentiment_df['video_id'] = self.sentiment_df['url'].apply(lambda x: video_id_from_url(x))
    self.sentiment_df['sentiment_topics'] = list(pos_neg_sentiment_topics.values())
    self.final_evaluation_df_extractor()


  def transcription_english_checker(self):  
    for chunk in self.complete_content_chunks:
      self.clean_transcript.append(cleanup_transcript(chunk))
    self.clean_transcript =  ''.join(self.clean_transcript)
    self.transcription_entities_extractor()


  def transcription_validator(self):   
    for url,transcription_details in self.transcriptions.items():
      for time_stamp,dialogue in transcription_details.items():
        self.complete_content.append(dialogue)
        self.time_based_content.append(f"{time_stamp}: {dialogue}")
    self.complete_content = ' '.join(self.complete_content)
    self.time_based_content = '/n'.join(self.time_based_content)
    self.complete_content_chunks = split_into_chunks(self.complete_content, 2500)
    self.transcription_english_checker()
  
  def final_evaluation_df_extractor(self):
    ##Extracting required data
    url = self.topic_df['topics'].iloc[0]['URL']
    theme = self.topic_df['topics'].iloc[0]['Overall Theme']
    introduction = self.topic_df['topics'].iloc[0]['Introduction']
    conclusion = self.topic_df['topics'].iloc[0]['Conclusion']
    topic_of_discussion = self.topic_df['topics'].iloc[0]['Topics of Discussion']
    category = self.topic_df['topics'].iloc[0]['Category']
    num_of_speakers = self.topic_df['topics'].iloc[0]['Number of Speakers']
    keywords = self.topic_df['topics'].iloc[0]['Keywords']
    video_type = self.topic_df['topics'].iloc[0]['Type of Video']
    summary = self.topic_df['topics'].iloc[0]['Summary']
    tone = self.topic_df['topics'].iloc[0]['Tone']
    num_of_scenes = self.topic_df['topics'].iloc[0]['Number of Scenes']
    self.trending_by_category.loc[0, ['url', 'theme', 'introduction', 'conclusion', 'topic_of_discussion', 'category', 'num_of_speakers', 'keywords', 'video_type', 'summary', 'tone', 'num_of_scenes']] = [url, theme, introduction, conclusion, topic_of_discussion, category, num_of_speakers, keywords, video_type, summary, tone, num_of_scenes]

  def youtube_video_eval_trigger(self):
    self.transcription_validator()
    return [self.trending_by_category,self.sentiment_df]
