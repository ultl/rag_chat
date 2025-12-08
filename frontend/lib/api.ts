import type { ChatMessage, ChatStreamEvent, DocumentItem, Session } from './types'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, '') ?? ''

async function handleResponse(res: Response) {
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res
}

export async function fetchSessions(): Promise<Session[]> {
  const res = await handleResponse(await fetch(`${API_BASE}/api/sessions`))
  return res.json() as Promise<Session[]>
}

export async function createSession(title?: string | null): Promise<Session> {
  const res = await handleResponse(
    await fetch(`${API_BASE}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title ?? null })
    })
  )
  return res.json() as Promise<Session>
}

export async function renameSession(sessionId: string, title: string): Promise<Session> {
  const res = await handleResponse(
    await fetch(`${API_BASE}/api/sessions/${sessionId}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title })
    })
  )
  return res.json() as Promise<Session>
}

export async function deleteSession(sessionId: string): Promise<void> {
  await handleResponse(
    await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
      method: 'DELETE'
    })
  )
}

export async function fetchMessages(sessionId: string): Promise<ChatMessage[]> {
  const res = await handleResponse(await fetch(`${API_BASE}/api/sessions/${sessionId}/messages`))
  return res.json() as Promise<ChatMessage[]>
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const res = await handleResponse(await fetch(`${API_BASE}/api/documents`))
  return res.json() as Promise<DocumentItem[]>
}

export async function deleteDocument(documentId: string): Promise<void> {
  await handleResponse(await fetch(`${API_BASE}/api/documents/${documentId}`, { method: 'DELETE' }))
}

export async function uploadDocument(file: File): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  await handleResponse(
    await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: form
    })
  )
}

export async function streamChat(
  payload: { session_id?: string | null; message: string },
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const res = await handleResponse(
    await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  )
  if (!res.body) throw new Error('No stream body')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''
    for (const chunk of parts) {
      const line = chunk.trim()
      if (!line.startsWith('data:')) continue
      const payload = JSON.parse(line.replace('data:', '').trim()) as ChatStreamEvent
      onEvent(payload)
    }
  }
}
