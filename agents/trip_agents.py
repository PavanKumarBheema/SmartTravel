import crewai
from crewai import Agent, LLM
from tools.trip_tools import MyCustomDuckDuckGoTool
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

class TripAgents:
	def local_expert_agent(self):
		return Agent(
			role='Local Expert at this city',
			goal='Provide the BEST insights about the selected city',
			backstory="""A knowledgeable local guide with extensive information
			about the city, its attractions, and customs.""",
			tools=[MyCustomDuckDuckGoTool()],
			llm=LLM(
				model="ollama/mistral",
				base_url="http://localhost:11434"
			),
			verbose=True,
				allow_delegation=False,
				max_iter=4,
			)
	def travel_concierge_agent(self):
		return Agent(
			role='Amazing Travel Concierge',
			goal="""Create the most amazing travel itineraries with budget and
				packing suggestions for the city""",
            backstory="""Specialist in travel planning and logistics with
				decades of experience""",
			tools=[MyCustomDuckDuckGoTool()],
			llm=LLM(
				model="ollama/mistral",
				base_url="http://localhost:11434"
			),
			verbose=True,
			allow_delegation=False,
			max_iter=4,
			)
