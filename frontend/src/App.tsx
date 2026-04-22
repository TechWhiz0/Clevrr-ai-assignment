import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { MarkdownMessage } from "./MarkdownMessage";
import { sendChat, ChatTurn } from "./api";

function ThinkingRow() {
  return (
    <div className="msg-row assistant" aria-live="polite">
      <div className="avatar avatar-assistant" aria-hidden>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
      </div>
      <div className="msg-column">
        <div className="msg-meta">
          <span className="msg-name">Shopify Analyst</span>
          <span className="msg-badge">Thinking</span>
        </div>
        <div className="thinking-bubble">
          <span className="thinking-dots">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </span>
        </div>
      </div>
    </div>
  );
}

export function App() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [history, busy, error]);

  async function onSubmit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend) return;
    setError(null);
    const msg = input.trim();
    setInput("");
    const prior = history;
    setHistory([...prior, { role: "user", content: msg }]);
    setBusy(true);
    try {
      const reply = await sendChat(msg, prior);
      setHistory([...prior, { role: "user", content: msg }, { role: "assistant", content: reply }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSubmit();
    }
  }

  return (
    <div className="app-shell">
      <header className="chat-header">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="brand-text">
            <h1 className="brand-title">Shopify Store Analyst</h1>
            <p className="brand-sub">Orders, products, and customers for your store</p>
          </div>
        </div>
      </header>

      <main className="chat-thread">
        <div className="thread-inner">
          {history.length === 0 && !busy ? (
            <div className="empty-state">
              <p className="empty-title">What would you like to know?</p>
              <p className="empty-hint">Type your own question. Press Enter to send, Shift+Enter for a new line.</p>
            </div>
          ) : null}

          {history.map((m, idx) => (
            <div key={idx} className={`msg-row ${m.role}`}>
              {m.role === "assistant" ? (
                <div className="avatar avatar-assistant" aria-hidden>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                </div>
              ) : (
                <div className="avatar avatar-user" aria-hidden>
                  You
                </div>
              )}
              <div className="msg-column">
                <div className="msg-meta">
                  <span className="msg-name">{m.role === "user" ? "You" : "Shopify Analyst"}</span>
                </div>
                <div className={`bubble ${m.role}`}>
                  {m.role === "assistant" ? (
                    <div className="assistant-body">
                      <MarkdownMessage content={m.content} />
                    </div>
                  ) : (
                    <MarkdownMessage content={m.content} />
                  )}
                </div>
              </div>
            </div>
          ))}

          {busy ? <ThinkingRow /> : null}
          <div ref={endRef} className="thread-anchor" />
        </div>
      </main>

      {error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}

      <footer className="chat-footer">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void onSubmit(e);
          }}
        >
          <div className="composer-inner">
            <textarea
              ref={textareaRef}
              className="composer-input"
              placeholder="Message your store analyst…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              disabled={busy}
            />
            <button className="composer-send" type="submit" disabled={!canSend} aria-label="Send message">
              {busy ? (
                <span className="send-spinner" aria-hidden />
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              )}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}
