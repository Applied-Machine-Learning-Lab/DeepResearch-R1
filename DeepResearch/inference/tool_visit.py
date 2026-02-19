import json
import os
import requests
import trafilatura
import urllib3
from qwen_agent.tools.base import BaseTool, register_tool

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    description = 'Visit a web page and extract its content in Markdown format.'
    name = 'visit'
    
    def call(self, params: str, **kwargs) -> str:

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
                    url = params_json.get('url', '')
                except:
                    url = params
            elif isinstance(params, dict):
                url = params.get('url', '')
            else:
                url = str(params)
            
            if isinstance(url, list):
                url = url[0] if url else ""
            
            if isinstance(url, str):
                url = url.strip(" []'\"")
        except Exception as e:
            return f"Error parsing parameters: {e}"
        
        if not url or not url.startswith('http'):
            return f"Error: Invalid URL format: {url}"

        print(f"   [Visit] Fetching: {url} via proxy (isolated session)...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,current_image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        try:
            response = session.get(
                url, 
                headers=headers, 
                proxies=proxies, 
                timeout=30, 
                verify=False,
                allow_redirects=True 
            )
            
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding
            
            html = response.text

            result = trafilatura.extract(
                html, 
                include_comments=False, 
                include_tables=True, 
                output_format="markdown",
                favor_precision=True
            )

            if not result:
                return f"Error: Visited {url} but failed to extract text content."

            if len(result) > 25000:
                result = result[:25000] + "\n\n[Content Truncated...]"

            return f"URL: {url}\n\nContent:\n{result}"

        except Exception as e:
            return f"Error visiting {url}: {str(e)}"