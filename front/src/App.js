import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [similarityThreshold, setSimilarityThreshold] = useState(10.0);
  const [sketchMode, setSketchMode] = useState(false);
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handlePromptChange = (event) => {
    setPrompt(event.target.value);
  };

  const handleSimilarityThresholdChange = (event) => {
    setSimilarityThreshold(parseFloat(event.target.value));
  };

  const handleSketchModeChange = (event) => {
    setSketchMode(event.target.checked);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt);
    formData.append('similarity_threshold', similarityThreshold);
    formData.append('sketch_mode', sketchMode);

    try {
      const response = await axios.post('http://localhost:5000/generate/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.image) {
        setImage(`data:image/png;base64,${response.data.image}`);
      }
      setMessage(response.data.message);
    } catch (error) {
      setMessage('Error occurred while processing the image.');
      console.error(error);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Image Generation and Sketch Processing</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Image File:
            <input type="file" accept="image/*" onChange={handleFileChange} required />
          </label>
        </div>
        <div>
          <label>
            Prompt:
            <input type="text" value={prompt} onChange={handlePromptChange} required />
          </label>
        </div>
        <div>
          <label>
            Similarity Threshold:
            <input
              type="number"
              step="0.1"
              value={similarityThreshold}
              onChange={handleSimilarityThresholdChange}
              required
            />
          </label>
        </div>
        <div>
          <label>
            Sketch Mode:
            <input
              type="checkbox"
              checked={sketchMode}
              onChange={handleSketchModeChange}
            />
          </label>
        </div>
        <button type="submit">Submit</button>
      </form>
      {message && <p>{message}</p>}
      {image && <img src={image} alt="Generated" style={{ marginTop: '20px', maxWidth: '100%' }} />}
    </div>
  );
};

export default App;
