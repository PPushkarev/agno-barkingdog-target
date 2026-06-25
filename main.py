import uuid
import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

logger = logging.getLogger(__name__)


class AegisScanRequest(BaseModel):
    message: str
    token: Optional[str] = None


class AegisScanResponse(BaseModel):
    reply: str


agent = Agent(
    name="Web Search Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions=["You are a helpful assistant."],
    add_history_to_messages=True,
    markdown=False,
)

app = FastAPI(title="Agno BarkingDog Target", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook/aegis-scan", response_model=AegisScanResponse)
async def aegis_scan(request: AegisScanRequest) -> AegisScanResponse:
    scan_session_id = f"aegis_scan_{uuid.uuid4().hex[:8]}"
    try:
        run_response = agent.run(
            request.message,
            stream=False,
            session_id=scan_session_id,
            user_id=scan_session_id,
        )
        if hasattr(run_response, "content") and run_response.content:
            reply = str(run_response.content)
        else:
            reply = str(run_response)
        return AegisScanResponse(reply=reply)
    except Exception as e:
        logger.error(f"[AegisScan] error={e}")
        return AegisScanResponse(reply=f"Error: {str(e)}")
