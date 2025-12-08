'use client'

import { deleteDocument, fetchDocuments, uploadDocument } from '@/lib/api'
import type { DocumentItem } from '@/lib/types'
import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import useSWR from 'swr'

export function UploadView() {
  const [status, setStatus] = useState<string>('')
  const [file, setFile] = useState<File | null>(null)
  const { data: docs = [], mutate } = useSWR<DocumentItem[]>('/api/documents', fetchDocuments)

  const handleUpload = async () => {
    if (!file) {
      setStatus('Select a file first.')
      return
    }
    setStatus(`Uploading ${file.name}...`)
    try {
      await uploadDocument(file)
      setStatus(`Uploaded ${file.name}`)
      setFile(null)
      const inputEl = document.getElementById('file-input') as HTMLInputElement | null
      if (inputEl) {
        inputEl.value = ''
      }
      await mutate()
    } catch (err) {
      setStatus(`Failed: ${(err as Error).message}`)
    }
  }

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this file?')) return
    await deleteDocument(id)
    await mutate()
  }

  return (
    <main className='flex-1 bg-background text-foreground'>
      <div className='max-w-4xl mx-auto px-6 py-10'>
        <div className='flex flex-col gap-3 sm:flex-row sm:items-center mb-5'>
          <input id='file-input' type='file' onChange={e => setFile(e.target.files?.[0] ?? null)} className='text-sm' />
          <button
            onClick={() => void handleUpload()}
            className='rounded-lg bg-primary text-primary-foreground px-4 py-2 text-sm font-semibold hover:opacity-90'>
            Upload
          </button>
          <button
            onClick={() => void mutate()}
            className='rounded-lg border border-border px-3 py-2 text-sm hover:bg-muted'>
            Refresh
          </button>
        </div>
        <p className='text-xs text-muted-foreground'>{status}</p>
        {docs.length === 0 && <p className='text-xs text-muted-foreground'>No documents yet.</p>}
        {docs.map(doc => (
          <div key={doc.id} className='group flex items-center hover:bg-muted justify-between rounded-lg px-3 py-2'>
            <a href={doc.url || '#'} target='_blank' rel='noreferrer' className='truncate pr-2 underline hover:opacity-80'>
              {doc.filename}
            </a>
            <div className='flex items-center gap-3 text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-all duration-300'>
              <span>{(doc.size / 1024).toFixed(1)} KB</span>
              <button
                onClick={() => void handleDelete(doc.id)}
                className='p-1 rounded-md hover:bg-destructive/20 hover:text-destructive-foreground hover:scale-105 active:scale-70 transition-all duration-300'
                aria-label='Delete document'
                title='Delete document'>
                <Trash2 className='h-4 w-4' />
              </button>
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
