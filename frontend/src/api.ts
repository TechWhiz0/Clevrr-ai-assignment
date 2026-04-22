export type ChatRole = "user" | "assistant";

export type ChatTurn = { role: ChatRole; content: string };

const apiBase = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";

export async function sendChat(message: string, history: ChatTurn[]): Promise<string> {
  const res = await fetch(`${apiBase}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      history: history.map((h) => ({ role: h.role, content: h.content })),
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }
  const data = (await res.json()) as { reply: string };
  return data.reply;
}
