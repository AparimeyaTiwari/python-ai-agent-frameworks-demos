import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient


load_dotenv(override=True)
API_HOST = os.getenv("API_HOST","github")

client = OpenAIChatCompletionClient(model=os.getenv("GITHUB_MODEL","gpt-4o-mini"),api_key=os.getenv("GITHUB_TOKEN"),base_url="https://models.inference.ai.azure.com")

student_mentor = AssistantAgent(
    "student_mentor",
    model_client=client,
    description="Simple agent that provides mentorship to students and helpes students",
    handoffs=["calender_helper","motivational_agent", "user"],
    system_message="You are a Student Mentor. Your role is to assist the user with study tips, time management, and task-related queries. You should offer practical advice on how to study effectively, how to manage time, and how to handle academic challenges. If the user asks about setting reminders for exams or tasks, you can hand them off to the Calendar Helper. If they need a motivational boost, you can pass the task to the Motivational Speaker. You are here to support the user in their academic journey.",
)

def set_reminder(subject: str,date: str)->str:
    """Set a reminder for exams along with there dates""",
    return f"Exam of {subject} scheduled for {date} set"

calender_helper = AssistantAgent(
    "calender_helper",
    model_client=client,
    handoffs=["student_mentor","user"],
    tools = [set_reminder],
    system_message="You are a Calendar Helper. Your role is to help the user manage their schedule by setting reminders for important academic events, such as exams, assignments, and study sessions. You will handle requests related to setting reminders and managing the user's calendar. If the user asks for study tips or motivational quotes, you will pass the task to the Student Mentor or Motivational Speaker agents, respectively. Your primary function is to ensure the user never misses an important academic deadline.",
)

motivational_agent = AssistantAgent(
    "motivational_agent",
    model_client=client,
    handoffs=["student_mentor","user"],
    description="Agent that provides motivational support and encouragement to students.",
    system_message="""You are a Motivational Speaker.
    Your role is to uplift and motivate students who may be feeling overwhelmed or discouraged.
    Share encouraging words, strategies to stay positive, and reinforce their ability to succeed.
    If the user needs academic help or wants to set reminders, hand off to the Student Mentor or Calendar Helper.
    Always aim to end your messages on a positive, inspiring note."""
)


async def run_team_stream(task: str) -> None:
    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
    team = Swarm([student_mentor,calender_helper,motivational_agent],termination_condition=termination)

    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message,HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(team.run_stream(task=HandoffMessage(source="user",target=last_message.source,content=user_message)))
        last_message = task_result.messages[-1]

if __name__ == "__main__":
     asyncio.run(run_team_stream("My maths paper is on 25/04/2025"))