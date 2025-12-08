'use client'

import './globals.css'
import { ThemeProvider } from '@/components/ThemeProvider'
import { Sidebar } from '@/components/Sidebar'
import { usePathname, useRouter } from 'next/navigation'
import type { ReactNode } from 'react'

export default function RootLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()
  const currentSessionId =
    pathname?.startsWith('/chat/') && pathname.split('/').filter(Boolean)[1]
      ? pathname.split('/').filter(Boolean)[1]
      : null

  return (
    <html lang='en' suppressHydrationWarning>
      <body className='min-h-screen bg-background text-foreground'>
        <ThemeProvider>
          <div className='flex min-h-screen'>
            <Sidebar
              currentSessionId={currentSessionId}
              onNewChat={() => router.push('/')}
              onSelectSession={id => router.push(`/chat/${id}`)}
            />
            <main className='flex-1 min-h-screen overflow-y-auto'>{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
