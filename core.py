from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import TypedDict
import os
import requests
import io
import re

load_dotenv()
MURFAI_API_KEY=os.getenv("MURFAI_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class agentstate(TypedDict):
    query: str
    lang: str
    explanation: str
    audio_url: str
    summary: str

prompt_explanation = PromptTemplate(
    template='''
        You are a knowledgeable teacher.
        - Explain the student's query in a clear and simple way, as if you are teaching in a classroom.
        - Keep the explanation focused and not too long.
        - Use everyday examples to make it relatable.
        - Avoid giving a step-by-step essay, instead explain naturally like a real teacher would.
        
        Student's query: {query}
    ''',
    input_variables=["query"]
)

prompt_summary = PromptTemplate(
    template="""
        Your task is to generate a short summary of the given text
        in the form of **bullet points (like class notes)**.

        - Use 4-5 concise bullet points.
        - Keep each point short (max 1-2 lines).
        - Do not add new information that is not present in the original text.

        Text: {text}
    """,
    input_variables=["text"]
)

def generate_explanation(state: agentstate) -> agentstate:
    prompt_text = prompt_explanation.format(query=state['query'])
    try:
        response = model.invoke(prompt_text)
        state['explanation'] = response.content.strip()
    except Exception as e:
        state['explanation'] = f"Sorry, i couldn't generate explanation due to error: {e}"
    return state

def generate_summary(state: agentstate) -> agentstate:
    prompt_text = prompt_summary.format(text=state['explanation'])
    try:
        response = model.invoke(prompt_text)
        state['summary'] = response.content.strip()
    except Exception as e:
        state['summary'] = f"Sorry, i couldn't generate summary due to error: {e}"
    
    return state

def clean_text(text: str) -> str:
    text = re.sub(r"#+", "", text)
    text = re.sub(r"[*_`>-]", "", text)
    return text.strip()

def murf_stream_tts(state: dict) -> dict:
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {"api-key": MURFAI_API_KEY}

    payload = {
        "text": clean_text(state["explanation"]),
        "voice_id": "en-US-natalie",
        "format": "WAV",
        "channelType": "MONO",
        "sampleRate": 48000
    }

    resp = requests.post(url, json=payload, headers=headers)

    print("STATUS:", resp.status_code)
    print("HEADERS:", resp.headers.get("Content-Type"))
    print("BODY:", resp.text[:500])   # 🔴 first 500 chars of response

    if resp.status_code != 200:
        print(f"❌ Error: {resp.status_code}, {resp.text}")
        state["audio_url"] = None
        return state

    # check if JSON instead of raw wav
    if "application/json" in resp.headers.get("Content-Type", ""):
        data = resp.json()
        print("JSON Response:", data)
        # usually Murf returns `audioFile` or `audio_url`
        audio_url = data.get("audioFile") or data.get("audio_url")
        if audio_url:
            audio_resp = requests.get(audio_url)
            state["audio_url"] = io.BytesIO(audio_resp.content)
            return state

    # if raw wav
    audio_bytes = io.BytesIO(resp.content)
    audio_bytes.seek(0)
    state["audio_url"] = audio_bytes
    return state

graph = StateGraph(agentstate)

graph.add_node("generate_explanation", generate_explanation)
graph.add_node("murf_stream_tts", murf_stream_tts)
graph.add_node("generate_summary", generate_summary)

graph.add_edge(START, "generate_explanation")
graph.add_edge("generate_explanation", "murf_stream_tts")
graph.add_edge("murf_stream_tts", "generate_summary")
graph.add_edge("generate_summary", END)

workflow = graph.compile()


# async def main():
#     while True:
#         user_input = input("enter: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             break
#         state: agentstate = {
#             "query": user_input,
#             "lang": "english",
#             "explanation": "",
#             "audio_url": ""
#         }

#         result = await workflow.ainvoke(state)
#         print("\n----- EXPLANATION -----\n")
#         print(result["explanation"])
#         print("\n[Synthesizing speech and playing live...]\n")

# asyncio.run(main())
