from autogen import AssistantAgent, UserProxyAgent

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

# Create an assistant agent
# assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
# user_proxy = UserProxyAgent(
#     "user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False}
# )  # IMPORTANT: set to True to run code in docker, recommended
# user_proxy.initiate_chat(
#     assistant, message="Plot a chart of NVDA and TESLA stock price change YTD."
# )
# # This initiates an automated chat between the two agents to solve the task

# response = oai.ChatCompletion.create(config_list)
# print(response)

# create an ai AssistantAgent named "assistant"
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

# create a human UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "agents-workspace",  # set the working directory for the agents to create files and execute
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""Create a fictious log of daily prices for TSLA stock from the month of December 2023 in a .csv file.""",
)
