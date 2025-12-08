from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.agent import AgentDeps, agent, build_history
from app.database import session_scope
from app.deps import get_embedder, get_redis_client, get_vector_store
from app.retrieval import Retriever


async def run_once(question: str) -> None:
  with session_scope() as session:
    retriever = Retriever(
      redis_client=get_redis_client(),
      session=session,
      vector_store=get_vector_store(),
      embedder=get_embedder(),
    )
    history = build_history([])

    async def _emit_stream(text: str, chunk_size: int = 48) -> None:
      for idx in range(0, len(text), chunk_size):
        piece = text[idx : idx + chunk_size]
        print(piece, end='', flush=True)
        await asyncio.sleep(0.02)

    async with agent.iter(question, deps=AgentDeps(retriever=retriever), message_history=history) as run:
      printed = False
      print()
      async for node in run:
        # Model responses (LLM outputs, possibly with tool calls)
        if hasattr(node, 'model_response'):
          for part in getattr(node.model_response, 'parts', []):
            if hasattr(part, 'tool_name'):
              kind = 'RETURN' if part.__class__.__name__ == 'ToolReturnPart' else 'CALL'
              print(f'{kind} :: {getattr(part, "tool_name", "")}')
              if hasattr(part, 'args'):
                print(f'  args: {getattr(part, "args", "")}')
              if hasattr(part, 'content'):
                print(f'  content: {getattr(part, "content", "")}')
            elif hasattr(part, 'content'):
              chunk = str(getattr(part, 'content', '') or '')
              if chunk:
                printed = True
                await _emit_stream(chunk)
        # Model requests carrying tool returns
        if hasattr(node, 'request'):
          for part in getattr(node.request, 'parts', []):
            if hasattr(part, 'tool_name'):
              kind = 'RETURN' if part.__class__.__name__ == 'ToolReturnPart' else 'CALL'
              print(f'{kind} :: {getattr(part, "tool_name", "")}')
              if hasattr(part, 'args'):
                print(f'  args: {getattr(part, "args", "")}')
              if hasattr(part, 'content'):
                print(f'  content: {getattr(part, "content", "")}')
      if not printed and run.result and run.result.output:
        await _emit_stream(str(run.result.output))
      print('\n--- end ---')


def main() -> None:
  print('CLI agent. Press Ctrl+C to exit.')
  while True:
    try:
      question = input('\nYou: ').strip()
    except (KeyboardInterrupt, EOFError):
      print('\nBye')
      break
    if not question:
      continue
    asyncio.run(run_once(question))


if __name__ == '__main__':
  main()
