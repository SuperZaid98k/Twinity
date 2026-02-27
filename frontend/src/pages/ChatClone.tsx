import { FormEvent, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Header from "@/components/Header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { chatWithClone, clearCloneChat } from "@/lib/api";
import { ArrowLeft, Loader2 } from "lucide-react";
import { toast } from "sonner";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatDocument = {
  filename?: string;
  unique_filename?: string;
  size?: number;
  description?: string;
};

const ChatClone = () => {
  const { cloneId } = useParams<{ cloneId: string }>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<ChatDocument[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const canSend = useMemo(() => !!cloneId && !!input.trim() && !isSending, [cloneId, input, isSending]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!canSend || !cloneId) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsSending(true);

    try {
      const response = await chatWithClone(cloneId, userMessage);
      setMessages((prev) => [...prev, { role: "assistant", content: response.response }]);
      setDocuments(response.documents || []);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to send message";
      toast.error(message);
    } finally {
      setIsSending(false);
    }
  };

  const handleClear = async () => {
    if (!cloneId) return;
    try {
      await clearCloneChat(cloneId);
      setMessages([]);
      toast.success("Chat cleared");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to clear chat";
      toast.error(message);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-16 max-w-4xl">
        <div className="flex items-center justify-between gap-3 mb-6">
          <Link to="/explore" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4" />
            Back to Explore
          </Link>
          <Button variant="outline" size="sm" onClick={handleClear} disabled={messages.length === 0}>
            Clear Chat
          </Button>
        </div>

        <div className="glass rounded-xl border border-border/50 p-4 md:p-6 min-h-[420px] flex flex-col">
          <div className="flex-1 space-y-3 overflow-y-auto pr-1">
            {messages.length === 0 && (
              <div className="text-sm text-muted-foreground">
                Start a conversation with this public clone.
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "ml-auto bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                }`}
              >
                {msg.content}
              </div>
            ))}
          </div>

          {documents.length > 0 && (
            <div className="mt-4 pt-4 border-t border-border/50">
              <p className="text-xs text-muted-foreground mb-2">Related documents</p>
              <div className="space-y-1">
                {documents.map((doc, idx) => {
                  const canDownload = !!doc.unique_filename;
                  const href = canDownload ? `/download_document/${cloneId}/${doc.unique_filename}` : undefined;
                  return (
                    <div key={`${doc.unique_filename || doc.filename || "doc"}-${idx}`}>
                      {canDownload ? (
                        <a className="text-sm text-primary hover:underline" href={href} target="_blank" rel="noreferrer">
                          {doc.filename || doc.unique_filename}
                        </a>
                      ) : (
                        <span className="text-sm text-foreground">{doc.filename || "Document"}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="pt-4 mt-4 border-t border-border/50 flex items-center gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask something..."
              className="bg-muted/50 border-border focus:border-primary"
            />
            <Button type="submit" disabled={!canSend} className="min-w-24">
              {isSending ? <Loader2 className="w-4 h-4 animate-spin" /> : "Send"}
            </Button>
          </form>
        </div>
      </main>
    </div>
  );
};

export default ChatClone;
