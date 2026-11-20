import { useState } from "react";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi! Describe your issue. I can help you understand possible legal options." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();
    const userMessage = input.trim();
    if (!userMessage) return;

    setMessages((old) => [...old, { role: "user", text: userMessage }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage })
      });
      const data = await response.json();
      setMessages((old) => [...old, {
        role: "bot",
        text: data.reply,
        links: data.links,
        urgency: data.urgency,
        intent: data.intent
      }]);
    } catch {
      setMessages((old) => [...old, {
        role: "bot",
        text: "Sorry, I could not connect to the server. Please make sure the backend is running."
      }]);
    }

    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: 20 }}>
      <h2>Know Your Rights Chatbot</h2>
      <div style={{ border: "1px solid #ccc", borderRadius: 10, padding: 15, height: 450, overflowY: "auto", marginBottom: 10, background: "#fff" }}>
        {messages.map((message, index) => (
          <div key={index} style={{ textAlign: message.role === "user" ? "right" : "left", marginBottom: 12 }}>
            <div style={{ display: "inline-block", padding: "10px 14px", borderRadius: 12, background: message.role === "user" ? "#007bff" : "#f1f1f1", color: message.role === "user" ? "#fff" : "#000", maxWidth: "80%", textAlign: "left" }}>
              <p style={{ margin: 0 }}>{message.text}</p>
              {message.links?.map((link) => (
                <a key={link} href={link} target="_blank" rel="noreferrer" style={{ display: "block", marginTop: 6, color: "#0066cc" }}>{link}</a>
              ))}
            </div>
          </div>
        ))}
        {loading && <p>Bot is typing...</p>}
      </div>
      <form onSubmit={sendMessage} style={{ display: "flex", gap: 10 }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Describe your issue..." style={{ flex: 1, padding: 10, borderRadius: 8, border: "1px solid #ccc" }} />
        <button type="submit" disabled={loading} style={{ padding: "10px 18px", borderRadius: 8, border: "none", background: "#007bff", color: "#fff", cursor: "pointer" }}>Send</button>
      </form>
    </div>
  );
}
