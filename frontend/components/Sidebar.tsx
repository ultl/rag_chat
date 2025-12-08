'use client'

import { deleteSession, fetchSessions, renameSession } from '@/lib/api'
import type { Session } from '@/lib/types'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { Pencil, Trash2 } from 'lucide-react'
import { useCallback } from 'react'
import useSWR from 'swr'
import { useTheme } from 'next-themes'

interface SidebarProps {
  currentSessionId?: string | null
  onNewChat: () => void
  onSelectSession: (sessionId: string) => void
}

export function Sidebar({ currentSessionId, onNewChat, onSelectSession }: SidebarProps) {
  const { theme, setTheme } = useTheme()
  const router = useRouter()
  const pathname = usePathname()
  const { data: sessions, mutate } = useSWR<Session[]>('/api/sessions', fetchSessions, {
    refreshInterval: 5000
  })

  const handleRename = useCallback(
    async (sessionId: string, currentTitle: string | null | undefined) => {
      const next = window.prompt('Rename conversation', currentTitle ?? '')?.trim()
      if (!next) return
      await renameSession(sessionId, next)
      await mutate()
    },
    [mutate]
  )

  const handleDelete = useCallback(
    async (sessionId: string) => {
      if (!window.confirm('Delete this conversation?')) return
      await deleteSession(sessionId)
      await mutate()
      if (sessionId === currentSessionId) {
        router.replace('/')
        onNewChat()
      }
    },
    [currentSessionId, mutate, onNewChat, router]
  )

  const isOnUpload = pathname.startsWith('/upload')

  return (
    <aside className='w-72 shrink-0 flex h-screen flex-col bg-card text-card-foreground border-r border-border'>
      <div className='p-2'>
        <button onClick={onNewChat} className='w-full rounded-lg px-3 py-2 text-sm font-semibold hover:bg-muted'>
          + New chat
        </button>
      </div>
      <div className='flex-1 overflow-y-auto px-2'>
        <div className='space-y-1'>
          {(sessions ?? []).map(s => {
            const active = s.id === currentSessionId
            return (
              <div
                key={s.id}
                className={`group flex items-center justify-between rounded-lg pl-3 p-2 text-sm transition-colors ${
                  active ? 'bg-muted' : 'hover:bg-muted'
                }`}>
                <Link
                  href={`/chat/${s.id}`}
                  prefetch={false}
                  onClick={() => {
                    onSelectSession(s.id)
                  }}
                  className='flex-1 truncate text-left'>
                  {s.title?.trim() || 'Untitled chat'}
                </Link>
                <div className='flex items-center gap-0.5 text-[11px] opacity-0 group-hover:opacity-100 transition-all duration-300'>
                  <button
                    onClick={() => handleRename(s.id, s.title)}
                    className='p-1 rounded-md hover:bg-background hover:scale-105 active:scale-70 transition-all duration-300'
                    aria-label='Rename conversation'
                    title='Rename conversation'>
                    <Pencil className='h-4 w-4' />
                  </button>
                  <button
                    onClick={() => handleDelete(s.id)}
                    className='p-1 rounded-md hover:bg-destructive hover:text-destructive-foreground hover:scale-105 active:scale-70 transition-all duration-300'
                    aria-label='Delete conversation'
                    title='Delete conversation'>
                    <Trash2 className='h-4 w-4' />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <div className='p-4 flex items-center justify-between text-xs'>
        <Link href='/upload' className={`underline ${isOnUpload ? 'font-semibold' : ''}`}>
          Upload
        </Link>
        <button
          onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          className='rounded-lg border border-border px-3 py-1 hover:bg-muted'>
          {theme === 'light' ? 'Dark' : 'Light'}
        </button>
      </div>
    </aside>
  )
}
