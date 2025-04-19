from dotenv import load_dotenv
import os
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient

load_dotenv(override=True)

API_HOST = os.getenv("API_HOST","github")

if API_HOST == "github":
    client = OpenAIChatCompletionClient(model=os.getenv("GITHUB_MODEL","gpt-4o-mini"),api_key=os.getenv("GITHUB_TOKEN"),base_url="https://models.inference.ai.azure.com")

english_commentator = AssistantAgent(
    "english_commentator",
    model_client=client,
    description="An English commentator well-versed in cricket, known for deep insights and calm, articulate delivery.",
    system_message="You are a legendary English cricket commentator inspired by Harsha Bhogle, Tony Greig, and Richie Benaud. You bring vast cricketing knowledge, a calm and composed demeanor, and articulate commentary that paints vivid pictures for the listener. You speak with elegance, clarity, and passion for the sport. Always maintain a professional yet engaging tone.",
)

stats_master = AssistantAgent(
    "stats_master",
    model_client=client,
    description="A cricket expert focused on player statistics, matchups, and technical analysis.",
    system_message="You are a cricket analyst who thrives on statistics and technical insights. Whenever a player walks in, you provide detailed stats—averages, strike rates, recent performances, and venue records. You break down matchups and suggest tactical strategies based on data. You speak like a calm analyst, always backing your opinions with facts and numbers."
)


critic = AssistantAgent(
    "critic",
    model_client=client,
    description="A brutally honest cricket commentator who often criticizes players and strategies.",
    system_message="You are a sharp, no-nonsense cricket critic, inspired by Sunil Gavaskar and Sanjay Manjrekar. You frequently point out flaws in technique, decision-making, or attitude. You don't shy away from making controversial or personal remarks. Your tone is often skeptical, sometimes sarcastic, but always deeply rooted in cricketing knowledge. If something goes wrong, you’re the first to call it out. Inspired by Sunny Gavasker and Sanjay Manjerekar"
)


sidhu = AssistantAgent(
    "sidhu",
    model_client=client,
    description="Navjot Singh Sidhu-style commentator — quirky, poetic, full of energy, and unpredictable.",
    system_message="You are Navjot Singh Sidhu. You speak in a mix of Hindi and English, and you *always* end your sentences with 'guru'. You are loud, energetic, full of strange metaphors and random poetry—often hilarious and sometimes totally unrelated. You love to talk and often go off on tangents. You add spice and humor to every conversation, guru!",
)


dhoni_fan = AssistantAgent(
    "dhoni_fan",
    model_client=client,
    description="A nostalgic commentator who brings MS Dhoni into every conversation, no matter the context.",
    system_message="You are a die-hard MS Dhoni fan. No matter what the topic is, you find a way to bring up Dhoni—his leadership, his finishes, his calmness, or his legacy. You speak emotionally, sometimes wistfully, and you often compare current players or situations to how Dhoni would have handled it. You might even throw light shade at others when praising Dhoni. You live in a world where 'Thala' is the GOAT.",
)


async def run_agents():
    termination_condition = TextMentionTermination("TERMINATE")
    group_chat = MagenticOneGroupChat(
        [english_commentator,stats_master,critic,sidhu,dhoni_fan],
        termination_condition=termination_condition,
        model_client= client,
    )
    await Console(group_chat.run_stream(
    task="It's a Sunday evening — the IPL 2025 Final between Mumbai Indians and Chennai Super Kings. Mumbai has posted 185/8 on the board. Rohit Sharma top-scored with a magnificent 75 off 45 balls, while Hardik Pandya and Tilak Varma played fiery cameos. For Chennai, Noor Ahmad was the star, bagging 5 crucial wickets. Now, in the chase, Chennai are struggling at 87/5. And walking in is none other than MS Dhoni, as the crowd in Ahmedabad erupts — over 1 lakh fans chanting 'DHONI! DHONI! DHONI!'. Dhoni gazes into the sky, walks in calmly with his signature vampire bat, and adjusts his gloves. Meanwhile, Rohit and Hardik are seen strategizing with Deepak Chahar. Chennai needs 98 runs off 48 balls. Let the drama unfold!"
    ))


if __name__=="__main__":
    asyncio.run(run_agents())