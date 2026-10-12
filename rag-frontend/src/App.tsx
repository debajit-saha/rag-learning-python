
import { useState } from "react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);

  async function askQuestion() {
    const currentQuestion = question.trim();

    if (!currentQuestion || loading) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", content: currentQuestion },
    ]);
    setQuestion("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, conversation_id: conversationId, }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data: {
        answer: string;
        conversation_id: string;
      } = await response.json();

      setConversationId(data.conversation_id);

      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to contact the AI service."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Employee Handbook Assistant</h1>
      <p>Ask questions about company policies and benefits.</p>

      <section className="chat">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role}`}
          >
            <strong>
              {message.role === "user" ? "You" : "Assistant"}
            </strong>
            <p>{message.content}</p>
          </div>
        ))}

        {loading && <p className="status">Assistant is thinking...</p>}
      </section>

      {error && <p className="error">{error}</p>}

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void askQuestion();
        }}
      >
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about annual leave, benefits..."
          rows={3}
          disabled={loading}
        />

        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </main>
  );
}

export default App;
