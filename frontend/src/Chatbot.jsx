import { useState } from "react";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Namaste! 🇮🇳 I am your Know Your Rights assistant. Describe your issue, and I will help you understand possible legal-awareness options."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();

    const userMessage = input.trim();
    if (!userMessage) return;

    setMessages((old) => [
      ...old,
      { role: "user", text: userMessage }
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch("https://know-your-rights-backend-0sg2.onrender.com/chat", {
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: userMessage
        })
      });

      const data = await response.json();

      setMessages((old) => [
        ...old,
        {
          role: "bot",
          text: data.reply,
          links: data.links,
          urgency: data.urgency,
          intent: data.intent
        }
      ]);
    } catch {
      setMessages((old) => [
        ...old,
        {
          role: "bot",
          text: "Sorry, I could not connect to the server. Please make sure the backend is running."
        }
      ]);
    }

    setLoading(false);
  }

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="header-icon">🇮🇳</div>

        <div>
          <h1>Know Your Rights</h1>
          <p>Understand your rights. Protect yourself.</p>
        </div>
      </header>

      <div className="chat-box">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message-row ${message.role === "user" ? "user-row" : "bot-row"}`}
          >
            <div className={`message ${message.role === "user" ? "user-message" : "bot-message"}`}>
              <p>{message.text}</p>

              {message.links && message.links.length > 0 && (
                <div className="links">
                  {message.links.map((link) => (
                    <a key={link} href={link} target="_blank" rel="noreferrer">
                      Official resource ↗
                    </a>
                  ))}
                </div>
              )}

              {message.urgency === "critical" && (
                <div className="urgent-note">
                  Immediate danger? Call 112 in India.
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && <p className="typing">Legal assistant is thinking...</p>}
      </div>

      <form className="chat-form" onSubmit={sendMessage}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe your issue..."
          className="chat-input"
        />

        <button type="submit" disabled={loading} className="send-button">
          ⚖️
        </button>
      </form>

      <p className="disclaimer">
        General legal awareness only. Not legal advice. For emergencies in India, call 112.
      </p>
    </div>
  );
}