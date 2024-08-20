import pandas as pd


def get_trending_videos(category_id, max_results):
  trending_by_category = pd.DataFrame(columns=['category','title',
                                               'viewCount','publishDate',
                                               'video_id','url'])
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

def youtube_video_categories_extractor(api_key):
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

api_key = 'AIzaSyBRvAdrZonkE--IGTqU22So6Dnt-6xfJg0'
categories_df = youtube_video_categories_extractor(api_key)
trending_by_category = pd.DataFrame(columns=['category','title','viewCount','publishDate','video_id','url'])
#for category_id in [1,2,10,15,17,20,22,24,25,26,28]:
for category_id in [22,25,26,28]:
  max_results = 10    # Number of videos to fetch
  trending_by_category = pd.concat([trending_by_category,get_trending_videos(category_id, max_results)])
