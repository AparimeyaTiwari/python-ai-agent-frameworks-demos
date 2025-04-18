import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_core import CancellationToken
import azure.identity

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST","github")
if API_HOST == "github":
    client = OpenAIChatCompletionClient(model=os.getenv("GITHUB_MODEL","gpt-4o-mini"),api_key=os.getenv("GITHUB_TOKEN"),base_url="https://models.inference.ai.azure.com")
elif API_HOST=="azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAIChatCompletionClient(model=os.environ["AZURE_OPENAI_CHAT_MODEL"], api_version=os.environ["AZURE_OPENAI_VERSION"], azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_ad_token_provider=token_provider)

agent = AssistantAgent(
    name="french_tutor",
    model_client=client,
    system_message="You are a french tutor who helps people learn french. Respond in french and also provide translation to english.",
)

async def main()->None:
    response = await agent.on_messages(
        [TextMessage(content="Hello their? How are you doing today?",source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)

if __name__=="__main__":
    asyncio.run(main())
