export type Theme = 'light' | 'dark'

export interface Session {
  id: string
  title?: string | null
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  id?: string
  role: 'user' | 'assistant'
  content: string
  created_at?: string
  documents?: DocumentRef[]
  support?: boolean
  tools?: string[]
  chunks?: ChunkRef[]
  toolLogs?: ToolLog[]
  tool_logs?: ToolLog[]
}

export interface DocumentRef {
  id: string
  filename: string
  url: string
}

export interface ChunkRef {
  chunk_id: string
  text: string
  document_id?: string
}

export interface ToolLog {
  kind: 'call' | 'return'
  tool: string
  args?: unknown
  content?: unknown
}

export type ChatStreamEvent =
  | { type: 'delta'; token: string }
  | {
      type: 'final'
      session_id: string
      text: string
      documents: DocumentRef[]
      support: boolean
      tools: string[]
      chunks: ChunkRef[]
      tool_logs?: ToolLog[]
    }
  | { type: 'tool'; logs: ToolLog[]; doc_ids: string[]; chunks: ChunkRef[] }

export interface DocumentItem {
  id: string
  filename: string
  size: number
  created_at: string
  url: string
}
