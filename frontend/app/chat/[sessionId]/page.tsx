import { ChatView } from '@/components/ChatView'

interface Props {
  params: { sessionId: string }
}

export default function ChatWithSession({ params }: Props) {
  return <ChatView sessionId={params.sessionId} />
}
