import os
import sys
from typing import List, Any, Dict
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
import googlemaps
from config import GOOGLE_MAPS_API, OPEN_TRIP_MAP_API, OPENAI_API_KEY


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = OpenAI(temperature=0.7)

# 1. Set up Google Maps and Open Trip Map API

maps = googlemaps.Client(GOOGLE_MAPS_API)

#['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_generate_auth_url', '_get', '_get_body', '_request', 'addressvalidation', 'base_url', 'channel', 'clear_experience_id', 'client_id', 'client_secret', 'directions', 'distance_matrix', 'elevation', 'elevation_along_path', 'find_place', 'geocode', 'geolocate', 'get_experience_id', 'key', 'nearest_roads', 'place', 'places', 'places_autocomplete', 'places_autocomplete_query', 'places_nearby', 'places_photo', 'queries_per_minute', 'queries_per_second', 'queries_quota', 'requests_kwargs', 'retry_over_query_limit', 'retry_timeout', 'reverse_geocode', 'sent_times', 'session', 'set_experience_id', 'snap_to_roads', 'snapped_speed_limits', 'speed_limits', 'static_map', 'timeout', 'timezone']

#maps.geocode("")


# 2. Define class and tools openai can use

class MapsTools:
    @staticmethod
    def recommend_places(place):
        prompt=f"Based on the {place} the user wants to visit, find and recommend 5 popular places (restaurants, tourist attractions, etc.) that are near."
        return llm.invoke(prompt).strip().split('\n')
    
    @staticmethod
    def plan(places, days):
        prompt=f"Based on the {places} the user wants to visit and how many {days} they plan to stay, make a travel plan."
        return llm.invoke(prompt)


# 3. Define promt template


# 4. Main program loop

# Define tools
tools = [
    Tool(
        func=MapsTools.recommend_places,
        name="RecommendPlaces",
        description="Find and Recommend 5 Popular places near the place user wants to visit."
    ),
    Tool(
        func=MapsTools.plan,
        name="Plan",
        description="Plan a trip, according to the places user wants to visit and how many days they want to stay."
    )
]

def get_prompt_template() -> PromptTemplate:
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You are a travel planner assistant. The user will input a city or country they want to visit and how long they will stay for.
Then, based on their needs, use {tools} to recommend places to visit and plan their trip.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

# Initialize the agent
prompt = get_prompt_template()
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", input_key="input")

# Create React agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# Main execution
def main():
    print("Welcome to your AI-powered Travel Assistant!")
    while True:
        user_input = input("What would you like to do? (Type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Music Assistant. Goodbye!")
            break
        
        # Prepare the input for the agent
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        
        response = agent_executor.invoke({
            "input": user_input,
            "tools": tool_strings,
            "tool_names": tool_names,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })
        print(f"Assistant: {response['output']}")

if __name__ == "__main__":
    main()