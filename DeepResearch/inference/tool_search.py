import json
import os
import requests
import time
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('search', allow_overwrite=True)
class Search(BaseTool):
    description = 'Perform web searches using Serper (Google Search API). Returns top results. Accepts multiple queries.'
    name = 'search'
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "The queries to perform searches on."
            }
        },
        "required": ["query"]
    }

    def call(self, params: str, **kwargs) -> str:
        serper_api_key = os.getenv("SERPER_KEY_ID")
        if not serper_api_key:
            return "Error: SERPER_KEY_ID not found in environment variables."

        session = requests.Session()
        session.trust_env = False  
        
        PROXY_URL = ""
        proxies = {
            "http": PROXY_URL,
            "https": PROXY_URL
        }

        try:
            if isinstance(params, str):
                try:
                    params_json = json.loads(params)
                    queries = params_json.get('query', [])
                except:
                    queries = [params]
            elif isinstance(params, dict):
                queries = params.get('query', [])
            else:
                queries = [str(params)]
            
            if isinstance(queries, str):
                queries = [queries]
        except:
            return "Error: Invalid parameters format."

        results = []
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json',
            'Connection': 'close'
        }

        for query in queries:
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    print(f"   [Search] Querying Serper: {query} ...")
                    payload = {"q": query, "num": 8}
                    
                    response = session.post(
                        url, 
                        headers=headers, 
                        json=payload, 
                        proxies=proxies, 
                        verify=False,  
                        timeout=30
                    )
                    response.raise_for_status()
                    search_results = response.json()

                snippets = []
                if "organic" in search_results:
                    for item in search_results["organic"][:5]:
                        title = item.get("title", "No Title")
                        link = item.get("link", "No Link")
                        snippet = item.get("snippet", "No Snippet")
                        snippets.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
                
                if "knowledgeGraph" in search_results:
                    kg = search_results["knowledgeGraph"]
                    kg_info = f"[Knowledge Graph] {kg.get('title', 'Info')}: {kg.get('description', '')}"
                    snippets.insert(0, kg_info)

                if snippets:
                    results.append(f"Query: {query}\n" + "\n---\n".join(snippets))
                else:
                    results.append(f"Query: {query}\nResult: No relevant results found on Google.")

                time.sleep(1.0)

            except Exception as e:
                print(f"   [Search Error]: {e}")
                results.append(f"Query: {query} Error: {str(e)}")

        return "\n======\n".join(results)