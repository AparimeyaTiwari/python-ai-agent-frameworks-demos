import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv(override=True)

# Spotify credentials from .env file
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

# Initialize Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-library-read user-read-playback-state user-modify-playback-state"
))

# Tool to play song on Spotify
@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    # Search for the song
    result = sp.search(q=song, limit=1, type="track")
    
    if not result["tracks"]["items"]:
        return f"Song '{song}' not found on Spotify."

    track = result["tracks"]["items"][0]
    track_uri = track["uri"]

    # Start playback of the song on Spotify
    sp.start_playback(uris=[track_uri])
    return f"Successfully played '{song}' on Spotify!"

# Register the tool
tools = [play_song_on_spotify]
tool_node = ToolNode(tools)

# Setup the client to use either Azure OpenAI or GitHub Models
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(), 
        "https://cognitiveservices.azure.com/.default"
    )
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_ad_token_provider=token_provider,
    )
else:
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["GITHUB_TOKEN"]
    )

model = model.bind_tools(tools, parallel_tool_calls=False)

# Define the agent logic
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if last_message.tool_calls else "end"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the LangGraph workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "action",
    "end": END,
})
workflow.add_edge("action", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Get user input and run the app
user_query = input("Enter your song request: ")
input_message = HumanMessage(content=user_query)

config = {"configurable": {"thread_id": "1"}}
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()