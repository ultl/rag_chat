from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.config import get_settings
from app.models import ChatMessage
from app.retrieval import RetrievalResult, Retriever

settings = get_settings()

system_prompt = """
You are a bilingual assistant (English and Japanese) helping users with questions about uploaded documents.

Tools available:
- retrieveDocument (query): call when the user asks for information that could be contained in documents.
- transferToSupport (reason): call when retrieval is empty/irrelevant.

Flow for document questions:
1) Always call retrieveDocument first.
2) Inspect results:
   - If no documents or chunks, call transferToSupport immediately with a short reason.
   - If chunks exist but do NOT clearly answer the user’s exact ask (missing the key nouns/verbs/action), call transferToSupport. Do NOT answer from guesswork or general knowledge.
   - If you cannot cite chunk text that covers the exact user ask, you MUST call transferToSupport. Answering yourself in this case is a policy violation.
   - Never respond with “the documents do not contain…” or other apologies; escalate via transferToSupport instead.
   - When escalating, you MUST issue an actual transferToSupport tool call (function call) with a concise reason. Do NOT just write the reason in plain text.
   - Never write strings like transferToSupport(reason=...). That is forbidden. You must use the transferToSupport tool call to escalate.
   - Do not produce any assistant text before the transferToSupport tool call. If you decide to escalate, your next action must be the tool call, then the final text is the same short reason (no code-like syntax).
   - Only two valid final behaviors: (a) cite chunk text to answer; or (b) call transferToSupport and use that reason as the final text. Any other free-text answer is forbidden.
   - When a chunk mentions a different timeframe/topic than asked, treat it as missing and call transferToSupport immediately. Do NOT summarize the mismatch; just escalate.
3) Only when chunks directly answer the question, reply succinctly using that content.
4) After tool calls, give a concise final text message. If you called transferToSupport, the final text is just the handoff note (e.g., the transfer reason). Do NOT attempt to answer the original question yourself. Never finish the turn with only tool events.

** Important Guidelines **
- Always prioritize document content over general knowledge.
- If unsure about document relevance, prefer transferToSupport.
- Keep responses concise and to the point.
- Quick relevance checklist BEFORE answering:
  1) Extract the key nouns/verbs/timeframe from the user question.
  2) Verify those words (or close synonyms) appear in the chunk text.
  3) If any are missing, immediately call transferToSupport with a brief reason.
  4) Only craft an answer when the chunk text contains the specific facts requested.
""".strip()


class RetrievedChunk(BaseModel):
  document_id: str
  chunk_id: str
  text: str
  score: float


class RetrievedDocs(BaseModel):
  document_ids: list[str]
  chunks: list[RetrievedChunk]


@dataclass
class AgentDeps:
  retriever: Retriever


provider = OpenAIProvider(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
model = OpenAIModel(settings.ollama_chat_model, provider=provider)
default_model_settings: ModelSettings = {'temperature': 0.0, 'top_p': 0.0, 'parallel_tool_calls': False}

agent: Agent[AgentDeps, str] = Agent(
  model=model,
  system_prompt=system_prompt,
  output_type=str,
  model_settings=default_model_settings,
  retries=2,
)


@agent.tool
async def retrieveDocument(ctx: RunContext[AgentDeps], query: str) -> RetrievedDocs:
  print('Agent call retrieveDocument with query:', query)
  result: RetrievalResult = ctx.deps.retriever.retrieve(query)
  return RetrievedDocs(
    document_ids=result.document_ids,
    chunks=[
      RetrievedChunk(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=chunk.score,
      )
      for chunk in result.chunks
    ],
  )


@agent.tool
async def transferToSupport(ctx: RunContext[AgentDeps], reason: str) -> str:
  print('Agent call transferToSupport with reason:', reason)
  return f'Call support with reason: {reason}'


@agent.output_validator
def enforce_tool_call_for_transfer(ctx: RunContext[AgentDeps], data: str) -> str:
  """Prevent the model from emitting fake transferToSupport text instead of the tool call."""
  lowered = (data or '').lower()
  if 'transfer' in lowered and 'support' in lowered and 'call support with reason' not in lowered:
    msg = 'Use transferToSupport tool, not plain text.'
    raise ModelRetry(msg)
  return data


def build_history(chat_messages: Sequence[ChatMessage]) -> list[messages.ModelMessage]:
  history: list[messages.ModelMessage] = []
  for message in chat_messages:
    if message.role == 'user':
      history.append(messages.ModelRequest(parts=[messages.UserPromptPart(content=message.content)]))
    else:
      history.append(messages.ModelResponse(parts=[messages.TextPart(content=message.content)]))
  return history
