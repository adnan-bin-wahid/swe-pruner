import json
import httpx
import logging
from typing import Optional
from .goal_models import StructuredGoal

logger = logging.getLogger(__name__)

class LocalGoalGeneratorClient:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:11434/v1"):
        self.endpoint_url = endpoint_url

    async def generate_goal(
        self, 
        prompt: str, 
        endpoint_url: Optional[str] = None, 
        model: Optional[str] = None
    ) -> Optional[StructuredGoal]:
        """
        Queries the local instruction model (e.g., Qwen2.5-Coder-1.5B-Instruct) 
        running at an OpenAI-compatible endpoint.
        """
        url = endpoint_url or self.endpoint_url
        model_name = model or "qwen2.5-coder:1.5b-instruct-q4_k_m"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a code context goal synthesis model. Respond ONLY with a valid JSON object matching the requested schema."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    parsed = json.loads(content)
                    return StructuredGoal(**parsed)
                else:
                    logger.warning(f"Local LLM API error: Status {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Failed to query local LLM at {url}: {e}")
        return None
