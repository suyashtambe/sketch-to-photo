import React, { useState } from "react";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("sketch", file);

    try {
      const res = await fetch("http://localhost:5000/generate", {
        method: "POST",
        body: formData,
      });

      const blob = await res.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResult(imageUrl);
    } catch (err) {
      console.error("Fetch error:", err);
    }
  };

  return (
    <div className="wrapper">
      <h1 className="title">Sketch-to-Photo Generator</h1>

      <div className="App">
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button type="submit">Generate</button>
        </form>

        {result && (
          <div>
            <h4>Result:</h4>
            <img src={result} alt="Generated" width="256" />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
