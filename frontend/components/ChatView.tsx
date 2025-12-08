'use client'

import { fetchMessages, streamChat } from '@/lib/api'
import type { ChatMessage, ChatStreamEvent } from '@/lib/types'
import { Streamdown } from 'streamdown'
import { useParams, useRouter } from 'next/navigation'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

interface ChatViewProps {
  sessionId?: string | null
}

export function ChatView({ sessionId: sessionIdProp }: ChatViewProps) {
  const params = useParams()
  const router = useRouter()
  const [sessionId, setSessionId] = useState<string | null>(sessionIdProp ?? null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)

  const loadMessages = useCallback(async (id: string) => {
    try {
      const msgs = await fetchMessages(id)
      setMessages(msgs)
    } catch (err) {
      console.error(err)
      setMessages([])
    }
  }, [])

  const lastLoadedId = useRef<string | null>(null)
  useEffect(() => {
    const routeId = typeof params?.sessionId === 'string' ? params.sessionId : null
    const targetId = sessionIdProp ?? routeId
    if (!targetId) return
    setSessionId(targetId)
    if (lastLoadedId.current === targetId && messages.length > 0) return
    lastLoadedId.current = targetId
    void loadMessages(targetId)
  }, [loadMessages, params?.sessionId, sessionIdProp, messages.length])

  const bubbleStyle = useMemo(() => {
    const base = 'text-sm leading-relaxed break-words whitespace-pre-wrap'
    return {
      user: `${base} bg-primary text-primary-foreground ml-auto rounded-2xl px-4 py-2`,
      assistant: base
    }
  }, [])

  const appendMessage = useCallback((msg: ChatMessage) => {
    setMessages(prev => [...prev, msg])
  }, [])

  const updateAssistant = useCallback(
    (
      updater: (prev: string) => string,
      docs?: ChatMessage['documents'],
      tools?: string[],
      chunks?: ChatMessage['chunks'],
      toolLogs?: ChatMessage['toolLogs']
    ) => {
      setMessages(prev => {
        const next = [...prev]
        const lastIndex = next.length - 1
        if (lastIndex >= 0 && next[lastIndex].role === 'assistant') {
          const current = next[lastIndex]
          next[lastIndex] = {
            ...current,
            content: updater(current.content),
            documents: docs ?? current.documents,
            tools: tools ?? current.tools,
            chunks: chunks ?? current.chunks,
            toolLogs: toolLogs ?? current.toolLogs
          }
        } else {
          next.push({ role: 'assistant', content: updater(''), documents: docs, tools, chunks, toolLogs })
        }
        return next
      })
    },
    []
  )

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return
    setStreaming(true)
    setInput('')
    appendMessage({ role: 'user', content: text })

    try {
      await streamChat({ session_id: sessionId, message: text }, (event: ChatStreamEvent) => {
        if (event.type === 'delta') {
          updateAssistant(prev => prev + event.token)
        } else if (event.type === 'final') {
          setSessionId(event.session_id)
          updateAssistant(
            () => event.text,
            event.documents ?? [],
            event.tools ?? [],
            event.chunks ?? [],
            event.tool_logs as ChatMessage['toolLogs']
          )
          if (!sessionId || sessionId !== event.session_id) {
            router.replace(`/chat/${event.session_id}`)
          }
        } else if (event.type === 'tool') {
          updateAssistant(
            prev => prev,
            undefined,
            undefined,
            event.chunks as ChatMessage['chunks'],
            event.logs as ChatMessage['toolLogs']
          )
        }
      })
    } catch (err) {
      console.error(err)
      appendMessage({ role: 'assistant', content: `Error: ${(err as Error).message}` })
    } finally {
      setStreaming(false)
    }
  }, [appendMessage, input, router, sessionId, streaming, updateAssistant])

  return (
    <div className='flex flex-col min-h-screen max-h-screen bg-background text-foreground'>
      <div className='flex-1 overflow-y-auto p-4 space-y-3'>
        {messages.map((m, idx) => (
          <div key={idx} className='flex max-w-4xl gap-3 mx-auto'>
            <div className={m.role === 'user' ? bubbleStyle.user : bubbleStyle.assistant}>
              {m.role === 'assistant' ? <Streamdown>{m.content}</Streamdown> : m.content}
              {m.role === 'assistant' && m.tools && m.tools.length > 0 && (
                <div className='mt-2 text-[11px] text-muted-foreground'>Tools used: {m.tools.join(', ')}</div>
              )}
              {m.role === 'assistant' && (m.toolLogs || m.tool_logs)?.length ? (
                <div className='mt-2 text-[11px] text-muted-foreground space-y-1'>
                  <div className='font-semibold text-foreground'>Agent steps</div>
                  {(m.toolLogs || m.tool_logs || []).map((log, i) => (
                    <div key={i} className='rounded bg-muted px-2 py-1'>
                      <div className='font-medium text-foreground'>
                        {log.kind.toUpperCase()} :: {log.tool}
                      </div>
                      {log.args && (
                        <pre className='whitespace-pre-wrap text-[10px]'>{JSON.stringify(log.args, null, 2)}</pre>
                      )}
                      {log.content && (
                        <pre className='whitespace-pre-wrap text-[10px]'>{JSON.stringify(log.content, null, 2)}</pre>
                      )}
                    </div>
                  ))}
                </div>
              ) : null}
              {m.role === 'assistant' && m.documents && m.documents.length > 0 && (
                <div className='mt-3 space-y-1 text-xs text-muted-foreground'>
                  <div className='font-semibold text-foreground'>References</div>
                  {m.documents.map(doc => (
                    <a
                      key={doc.id}
                      href={doc.url || '#'}
                      target='_blank'
                      rel='noreferrer'
                      className='underline block truncate'>
                      {doc.filename || doc.id}
                    </a>
                  ))}
                </div>
              )}
              {m.role === 'assistant' && m.chunks && m.chunks.length > 0 && (
                <div className='mt-3 space-y-1 text-[11px] text-muted-foreground'>
                  <div className='font-semibold text-foreground'>Chunks</div>
                  <div className='flex flex-wrap gap-1'>
                    {m.chunks.map(chunk => (
                      <span
                        key={chunk.chunk_id}
                        title={chunk.text}
                        className='rounded bg-muted px-1 py-0.5 hover:bg-accent transition-colors cursor-help'>
                        {chunk.chunk_id.slice(0, 8)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className='flex items-center gap-3 p-3 bg-card text-card-foreground border-t border-border'>
        <textarea
          rows={2}
          value={input}
          placeholder='Message'
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void handleSend()
            }
          }}
          className='flex-1 bg-transparent outline-none text-sm'
        />
        <button
          onClick={() => void handleSend()}
          disabled={streaming}
          className='rounded-lg bg-primary text-primary-foreground px-4 py-2 text-sm font-semibold hover:opacity-90 disabled:opacity-60'>
          Send
        </button>
      </div>
    </div>
  )
}
