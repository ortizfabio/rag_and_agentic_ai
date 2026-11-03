from manual_tool_calling_agent import llm_with_tools
from youtube_tools import YouTubeTools

tool_mapping = {
    "get_thumbnails": YouTubeTools.GET_THUMBNAILS,
    "extract_video_id": YouTubeTools.EXTRACT_VIDEO_ID,
    "fetch_transcript": YouTubeTools.FETCH_TRANSCRIPT,
    "search_youtube": YouTubeTools.SEARCH_YOUTUBE,
    "get_full_metadata": YouTubeTools.GET_FULL_METADATA
}

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

from langchain_core.runnables import RunnablePassthrough, RunnableSerializable

def create_summarization_chain()-> RunnableSerializable[dict[str, str], str] :
    return  (
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
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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

# Usage
summarization_chain = create_summarization_chain()
result = summarization_chain.invoke({
    "query": "Summarize this YouTube video: https://www.youtube.com/watch?v=1bUy-1hGZpI"
})

print("Video Summary:\n", result)

query = {"query": "Get top 3 youtube videos in India and their metadata"}
try:
    result = summarization_chain.invoke(query)
    print("Video Summary:\n", result)
except Exception as e:
    print("Non-critical network error:", e)

print(result)

recursive_chain = RunnableLambda(_recursive_chain)

RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
universal_chain: Runnable[dict[str, str], list[BaseMessage]] = (
    RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
    | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
    | recursive_chain
)
