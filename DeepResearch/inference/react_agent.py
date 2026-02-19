import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, Any
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    Scholar(),            
    Visit(),              
    Search(),            
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    

    def call_server(self, msgs, planning_port, max_tries=10):

        openai_api_key = "" 
        openai_api_base = "" 

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        for attempt in range(max_tries):
            try:
                print(f"--- Calling Local vLLM (Port 6006), try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.llm_local_path,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.6,
                    max_tokens=8192,
                )
                return chat_response.choices[0].message.content.strip()
            except Exception as e:
                print(f"本地 vLLM 呼叫失败: {e}")
                time.sleep(1)
        return "Error"

    def count_tokens(self, messages):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained("")

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        return token_count

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        search_count = 0
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        
        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                result = {"question": question, "answer": answer, "messages": messages, "prediction": 'Timeout', "termination": 'Timeout', "search_count": search_count}
                return result

            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round} Content Preview: {content[:100]}...')
            
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call_str.lower():
                        if 'PythonInterpreter' in TOOL_MAP:
                            try:
                                code_raw = tool_call_str.split('<code>')[1].split('</code>')[0].strip()
                                result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                            except:
                                result = "[Python Interpreter Error]: Formatting error."
                        else:
                            result = "Python tool is not enabled. Please use Google Search."
                    else:
                        tool_call = json5.loads(tool_call_str)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        
                        if tool_name == 'search':
                            search_count += 1
                        
                        print(f"   >>> Calling Tool: {tool_name} ...")
                        result = self.custom_call_tool(tool_name, tool_args)

                except Exception as e:
                    print(f"   [Tool Error]: {e}")
                    result = f'Error: Tool execution failed. {str(e)}'
                
                result = "<tool_response>\n" + str(result) + "\n</tool_response>"
                messages.append({"role": "user", "content": result})

            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
                
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > 110 * 1024:
                messages[-1]['content'] = "Token limit reached. Please provide final answer format."
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content:
                     termination = 'token limit reached'
                     break
                else:
                     termination = 'token limit reached (no answer)'
                     break

        if '<answer>' in messages[-1]['content']:
            try:
                prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            except:
                prediction = messages[-1]['content']
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "search_count": search_count
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path=""))
                result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--enable_web_search", type=str, default="false")
    parser.add_argument("--enable_retriever", type=str, default="false")
    parser.add_argument("--enable_calculator", type=str, default="false")
    parser.add_argument("--enable_file_parser", type=str, default="false")
    args = parser.parse_args()

    llm_config = {
        "model": args.model_path,
        "generate_cfg": {"temperature": 0.85, "top_p": 0.95, "presence_penalty": 1.1}
    }
    
    agent = MultiTurnReactAgent(llm=llm_config)
    
    with open(args.dataset, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_file = os.path.join(args.output_dir, "")
    print(f"Starting inference... Output: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(lines):
            if not line.strip(): continue
            try:
                print(f"\n=== Processing Item {i+1} ===")
                item_data = json.loads(line)
                input_packet = {'item': item_data, 'planning_port':6006}
                result = agent._run(data=input_packet, model=args.model_path)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    print("Done!")