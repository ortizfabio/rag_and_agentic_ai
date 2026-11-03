# Suppress warnings

from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from youtube_tools import *

warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)

query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"

messages = [HumanMessage(content = query)]
print(messages)
response_1 = llm_with_tools.invoke(messages)
print(json.dumps(response_1.model_dump(),indent=4))

messages.append(response_1)
tool_calls_1: List[Dict] = response_1.tool_calls
print(tool_calls_1)

tool_name=tool_calls_1[0]['name']
print(tool_name)

#%%
print("tool_call_id="+ tool_calls_1[0]['id'])
print("args= "+str(tool_calls_1[0]['args']))

my_tool=get_video_tool(tool_calls_1[0]['name'])

video_transcript =my_tool.invoke(tool_calls_1[0]['args'])
print(video_transcript)
#%%
messages.append(ToolMessage(content = video_transcript, tool_call_id = tool_calls_1[0]['id']))

#%%
response_2 = llm_with_tools.invoke(messages)
print(json.dumps(response_2.model_dump(),indent=4))
messages.append(response_2)
tool_call_2 = response_2.tool_calls
print(len(tool_call_2))

fetch_transcript_tool_output = get_video_tool(tool_call_2[0]['name']).invoke(tool_call_2[0]['args'])
fetch_transcript_tool_output

#%%
messages.append(ToolMessage(content = fetch_transcript_tool_output, tool_call_id = tool_call_2[0]['id']))

summary = llm_with_tools.invoke(messages)

query_us = {"query": "Show top 3 US trending videos with metadata and thumbnails"}

try:

    from tool_chain import universal_chain

    response = universal_chain.invoke(query_us)
    print("\nUS Trending Videos:\n", response[-1])
except Exception as e:
    print("Non-critical network error while fetching US trending videos:", e)

youtube_url = "https://www.youtube.com/watch?v=U2tYuer4em4"
video_transcript = extract_video_id.run(youtube_url)
print(f"Extracted video ID: {video_transcript}")

video_metadata = get_full_metadata.run(youtube_url)
print(f"Retrieved metadata for: {video_metadata['title']}")

transcript = fetch_transcript.run(video_transcript)
print(f"Retrieved transcript with {len(transcript)} characters")

#%%
thumbnails = get_thumbnails.run(youtube_url)
print(f"Retrieved {len(thumbnails)} thumbnails")

prompt = f"""
Please analyze this YouTube video and provide a comprehensive summary.

VIDEO TITLE: {video_metadata['title']}
CHANNEL: {video_metadata['channel']}
VIEWS: {video_metadata['views']}
DURATION: {video_metadata['duration']} seconds
LIKES: {video_metadata['likes']}

TRANSCRIPT EXCERPT:
{transcript[:3000]}... (transcript truncated for brevity)

Based on this information, please provide:
1. A concise summary of the video content (3-5 bullet points)
2. The main topics or themes discussed
3. The intended audience for this content
4. A
"""

# Get video transcript
messages = [HumanMessage(content=prompt)]
response = llm_model.invoke(messages)

#%%
# Get video transcript
print("\n===== VIDEO ANALYSIS =====\n")
print(response.content)


