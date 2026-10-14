const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001";

export default function App() {
  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem" }}>
      <h1>Real-Time Multimodal Traffic Intelligence</h1>
      <p>Dashboard placeholder. API: {API_BASE_URL}</p>
    </main>
  );
}
