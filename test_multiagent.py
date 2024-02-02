from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

# BASE_URL = "http://localhost:11434/api"
# BASE_URL = "http://localhost:5234/chat"
BASE_URL = "http://localhost:8000/"

ollama_base_config = {
    "base_url": BASE_URL,
    "api_type": "open_ai",
    "api_key": "NULL",
}


def model_config(model_name, **kwargs):
    return {**ollama_base_config, "model": model_name, **kwargs}


config_list = [
    model_config("nous-hermes2:10.7b-solar-q5_K_M"),
    model_config("ollama run phi:2.7b-chat-v2-q5_K_M"),
]

# set a "universal" config for the agents
agent_config = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "request_timeout": 120,
}

# humans
user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)

# agents
engineer = AssistantAgent(
    name="Engineer",
    llm_config=agent_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)

scientist = AssistantAgent(
    name="Scientist",
    llm_config=agent_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

planner = AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=agent_config,
)

critic = AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=agent_config,
)

# start the "group chat" between agents and humans
groupchat = GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, critic],
    messages=[],
    max_round=50,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=agent_config)

# Start the Chat!
user_proxy.initiate_chat(
    manager,
    message="""
find papers on LLM applications from arxiv in the last week, create a markdown table of different domains. then save it as a markdown file.
""",
)
