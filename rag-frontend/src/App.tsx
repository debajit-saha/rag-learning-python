import { useState } from "react";
import "./App.css";

type RetrievedImage = {
  name: string;
  description: string;
  page: number;
  url: string;
  relevance_score?: number;
};

type Citation = {
  number: number;
  file_name: string;
  page: number;
  chunk: number;
  snippet: string;
  url: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  images?: RetrievedImage[];
  citations?: Citation[];
};

type AskResponse = {
  answer: string;
  conversation_id: string | null;
  images?: RetrievedImage[];
  citations?: Citation[];
};

const API_BASE_URL = "http://127.0.0.1:8000";

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
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: currentQuestion,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data: AskResponse = await response.json();

      setConversationId(data.conversation_id);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          images: data.images ?? [],
          citations: data.citations ?? [],
        },
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

            <p className="message-content">{message.content}</p>

            {message.role === "assistant" &&
              message.images &&
              message.images.length > 0 && (
                <div className="retrieved-images">
                  {message.images.map((image) => (
                    <figure
                      key={`${image.name}-${image.page}`}
                      className="retrieved-image"
                    >
                      <img
                        src={`${API_BASE_URL}${image.url}`}
                        alt={image.description}
                        loading="lazy"
                      />

                      <figcaption>
                        <strong>{image.name}</strong>
                        <p>{image.description}</p>
                        <span>Page {image.page}</span>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              )}

            {message.role === "assistant" &&
              message.citations &&
              message.citations.length > 0 && (
                <div className="citations">
                  <strong>Sources</strong>
                  <ol>
                    {message.citations.map((citation) => (
                      <li key={citation.number} value={citation.number}>
                        <a
                          href={`${API_BASE_URL}${citation.url}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {citation.file_name}, page {citation.page}
                        </a>
                        <p>{citation.snippet}</p>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
          </div>
        ))}

        {loading && (
          <p className="status">Assistant is thinking...</p>
        )}
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

        <button
          type="submit"
          disabled={loading || !question.trim()}
        >
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </main>
  );
}

export default App;