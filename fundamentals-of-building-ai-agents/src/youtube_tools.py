# Suppress warnings
import builtins
import json
import warnings
from enum import IntEnum
from typing import List, Dict, Any, cast, Callable, Sequence

import yt_dlp
from langchain_core.runnables import Runnable
from langchain_core.tools import tool, StructuredTool, BaseTool

warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging

pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)


@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 10-character YouTube video ID from a URL.

    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    import re
    # Regex pattern to match video IDs
#    pattern = r'(?:v=|be/|embed/)([a-zA-Z-1-9_-]{11})'
    pattern = r'(?:v=|be/|embed/|v%3D)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    response = match.group(1) if match else "Error: Invalid YouTube URL"
    return response


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
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube(query: str) -> list[dict[str, str | Any]] | str:
    """
    Search YouTube for videos matching the query.

    Args:
        query (str): The search term to look for on YouTube

    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    from pytube import Search
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


class YouTubeTools(IntEnum):
    """ This class is used to map tool names to indices in an array.
    The array should be filled using the order of the indexes here """
    EXTRACT_VIDEO_ID = 0
    GET_FULL_METADATA = 1
    FETCH_TRANSCRIPT = 2
    SEARCH_YOUTUBE = 3
    GET_THUMBNAILS = 4

    @property
    def function(self):
        """Maps the enum index to its corresponding raw python function."""
        mapping = {
            YouTubeTools.EXTRACT_VIDEO_ID: extract_video_id,
            YouTubeTools.GET_FULL_METADATA: get_full_metadata,
            YouTubeTools.FETCH_TRANSCRIPT: fetch_transcript,
            YouTubeTools.SEARCH_YOUTUBE: search_youtube,
            YouTubeTools.GET_THUMBNAILS: get_thumbnails,
        }
        return mapping[self]


def get_video_tools() -> Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool]:
    """
    This method returns an array of tools in an order that makes safe to call them
    if used with the indexes of the YouTubeTools enum. This is important
    for the agent to know which tool to call.
    This function returns a list of tools. The order of the tools needs
    to match the order of the indexes in the YouTubeTools enum. This is important
    for the agent to know which tool to call.
    """
    return [
        YouTubeTools.GET_THUMBNAILS.function,
        YouTubeTools.EXTRACT_VIDEO_ID.function,
        YouTubeTools.FETCH_TRANSCRIPT.function,
        YouTubeTools.SEARCH_YOUTUBE.function,
        YouTubeTools.GET_FULL_METADATA.function,
    ]

def get_video_tool(tool_name: str) -> StructuredTool | None:
    """
    This method returns a tool by its name. The name of the tool is the same as the function name.
    """
    tools = get_video_tools()
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None

from langchain_llm import get_llm
llm_model = get_llm("llama3.1:8b", model_provider="ollama")
tools: Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool]= get_video_tools()
llm_with_tools = llm_model.bind_tools(tools)


if __name__ == "__main__":

    video_url = "https://www.youtube.com/watch?v=T-D1OfcDW1M"
    for tool in tools:
        schema = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema.schema() if tool.args_schema else {},
            "return": tool.return_type if hasattr(tool, "return_type") else None}
        print(json.dumps(schema))

    video_id = tools[YouTubeTools.EXTRACT_VIDEO_ID.value].run(video_url)
    print("video_id: ", video_id)
    transcript = tools[YouTubeTools.FETCH_TRANSCRIPT.value].run(video_id)
    print("transcript:\n", transcript)
    search_result = tools[YouTubeTools.SEARCH_YOUTUBE.value].run("Generative AI")
    print(json.dumps(search_result))
    full_metadata = tools[YouTubeTools.GET_FULL_METADATA.value].run(video_url)
    print(json.dumps(full_metadata))
    thumbnails = tools[YouTubeTools.GET_THUMBNAILS.value].run(video_url)
    print(json.dumps(thumbnails))
