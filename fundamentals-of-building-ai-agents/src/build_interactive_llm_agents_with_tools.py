from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
import re
from pytube import YouTube
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)
from llamaindex_llm import simple_llm

llm = init_chat_model("llama3.1:8b", model_provider="ollama")


@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.

    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """

    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"

# print(extract_video_id.name)
# print("----------------------------")
# print(extract_video_id.description)
# print("----------------------------")
# print(extract_video_id.func)

extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A")

# print(extract_video_id)
tools = []
tools.append(extract_video_id)
from youtube_transcript_api import YouTubeTranscriptApi


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.

    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").

    Returns:
        str: The transcript text or an error message.
    """

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"

fetch_transcript.run("hfIUstzHs9A")
tools.append(fetch_transcript)

# %%
from pytube import Search
from langchain.tools import tool
from typing import List, Dict


@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.

    Args:
        query (str): The search term to look for on YouTube

    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_transcript,
                "url": f"https://youtu.be/{yt.video_transcript}"
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"


#%%
#search_out=search_youtube.run("Generative AI")
# print(json.dumps(search_out))
tools.append(search_youtube)

#%%
@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }

#%%
#meta_data=get_full_metadata.run("https://www.youtube.com/watch?v=T-D1OfcDW1M")
#print(json.dumps(meta_data))

tools.append(get_full_metadata)

# %%
@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.

    Args:
        url (str): YouTube video URL (any format)

    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """

    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)

            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })

            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]


#thumbnails = get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")

#print(json.dumps(thumbnails))
tools.append(get_thumbnails)
#%%
llm_with_tools = llm.bind_tools(tools)

# %%
# for tool in tools:
#     schema = {
#         "name": tool.name,
#         "description": tool.description,
#         "parameters": tool.args_schema.schema() if tool.args_schema else {},
#         "return": tool.return_type if hasattr(tool, "return_type") else None}
#     print(json.dumps(schema))


#%%
query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"
# print(query)
#%%
messages = [HumanMessage(content = query)]
print(messages)
response_1 = llm_with_tools.invoke(messages)
print(json.dumps(response_1,4))

messages.append(response_1)

tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

tool_calls_1 = response_1.tool_calls
print(json.dumps(tool_calls_1,4))

#%%
tool_name=tool_calls_1[0]['name']
print(tool_name)

#%%
tool_call_id =tool_calls_1[0]['id']
print(tool_call_id)
args=tool_calls_1[0]['args']
print(args)

my_tool=tool_mapping[tool_calls_1[0]['name']]

video_transcript =my_tool.invoke(tool_calls_1[0]['args'])
video_transcript
#%%
messages.append(ToolMessage(content = video_transcript, tool_call_id = tool_calls_1[0]['id']))

#%%
response_2 = llm_with_tools.invoke(messages)
print(json(response_2,4))
messages.append(response_2)
tool_call_2 = response_2.tool_calls
tool_call_2

fetch_transcript_tool_output = tool_mapping[tool_call_2[0]['name']].invoke(tool_call_2[0]['args'])
fetch_transcript_tool_output

#%%
messages.append(ToolMessage(content = fetch_transcript_tool_output, tool_call_id = tool_call_2[0]['id']))

summary = llm_with_tools.invoke(messages)


# Define the processing steps
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call["id"]
        )

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

#%%
summarization_chain = (
    # Start with initial query
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    # First LLM call (extract video ID)
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process first tool call
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc) for tc in x["ai_response"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    # Second LLM call (fetch transcript)
    | RunnablePassthrough.assign(
        ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process second tool call
    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    # Final message update
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    # Generate final summary
    | RunnablePassthrough.assign(
        summary=lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"])
)

# Usage
result = summarization_chain.invoke({
    "query": "Summarize this YouTube video: https://www.youtube.com/watch?v=1bUy-1hGZpI"
})

print("Video Summary:\n", result)

initial_setup = RunnablePassthrough.assign(
    messages=lambda x: [HumanMessage(content=x["query"])]
)

first_llm_call = RunnablePassthrough.assign(
    ai_response=lambda x: llm_with_tools.invoke(x["messages"])
)

#%%
first_tool_processing = RunnablePassthrough.assign(
    tool_messages=lambda x: [
        execute_tool(tc) for tc in x["ai_response"].tool_calls
    ]
).assign(
    messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
)

#%%
second_llm_call = RunnablePassthrough.assign(
    ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
)

#%%
second_tool_processing = RunnablePassthrough.assign(
    tool_messages2=lambda x: [
        execute_tool(tc) for tc in x["ai_response2"].tool_calls
    ]
).assign(
    messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
)

final_summary = RunnablePassthrough.assign(
    summary=lambda x: llm_with_tools.invoke(x["messages"]).content
) | RunnableLambda(lambda x: x["summary"])

#%% md
#### Assembling the complete chain



#%%
chain = (
    initial_setup
    | first_llm_call
    | first_tool_processing
    | second_llm_call
    | second_tool_processing
    | final_summary
)

query = {"query": "Get top 3 youtube videos in India and their metadata"}
try:
    result = summarization_chain.invoke(query)
    print("Video Summary:\n", result)
except Exception as e:
    print("Non-critical network error:", e)

print(result)

# %%
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage
import json


def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        content = f"Error: {str(e)}"

    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"]
    )


def process_tool_calls(messages):
    """Recursive tool call processor"""
    last_message = messages[-1]

    # Execute all tool calls in parallel
    tool_messages = [
        execute_tool(tc)
        for tc in getattr(last_message, 'tool_calls', [])
    ]

    # Add tool responses to message history
    updated_messages = messages + tool_messages

    # Get next LLM response
    next_ai_response = llm_with_tools.invoke(updated_messages)

    return updated_messages + [next_ai_response]

#%%
def should_continue(messages):
    """Check if you need another iteration"""
    last_message = messages[-1]
    return bool(getattr(last_message, 'tool_calls', None))

def _recursive_chain(messages):
    """Recursively process tool calls until completion"""
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages

recursive_chain = RunnableLambda(_recursive_chain)

RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
universal_chain = (
    RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
    | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
    | recursive_chain
)

query_us = {"query": "Show top 3 US trending videos with metadata and thumbnails"}

try:
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

#%%
# Get video transcript
messages = [HumanMessage(content=prompt)]
response = llm.invoke(messages)

#%%
# Get video transcript
print("\n===== VIDEO ANALYSIS =====\n")
print(response.content)


