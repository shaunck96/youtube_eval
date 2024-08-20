from utils import YoutubeVideoAIStatExtractor
import streamlit as st
import json

with open("./config.json", "r") as jsonfile:
    creds = json.load(jsonfile)
    print("Read successful")

trending_by_category = YoutubeVideoAIStatExtractor(url="https://www.youtube.com/watch?v=mrKuDK9dGlg")#,
                                                   #creds["youtube_api_key"],
                                                   #creds["open_ai_api_key"])
trending_by_category.youtube_video_eval_trigger()
