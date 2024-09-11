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
    const selectedFile = event.target.files[0];
    console.log('Selected file:', selectedFile);  // 콘솔 로그 추가
    setFile(selectedFile);
  };

  const handlePromptChange = (event) => {
    const newPrompt = event.target.value;
    console.log('Prompt:', newPrompt);  // 콘솔 로그 추가
    setPrompt(newPrompt);
  };

  const handleSimilarityThresholdChange = (event) => {
    const newThreshold = parseFloat(event.target.value);
    console.log('Similarity Threshold:', newThreshold);  // 콘솔 로그 추가
    setSimilarityThreshold(newThreshold);
  };

  const handleSketchModeChange = (event) => {
    const newSketchMode = event.target.checked;
    console.log('Sketch Mode:', newSketchMode);  // 콘솔 로그 추가
    setSketchMode(newSketchMode);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt);
    formData.append('similarity_threshold', similarityThreshold);
    formData.append('sketch_mode', sketchMode);

    console.log('Form Data:', {
      file,
      prompt,
      similarityThreshold,
      sketchMode,
    });  // 콘솔 로그 추가

    try {
      const response = await axios.post('http://localhost:5000/generate/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Response:', response);  // 콘솔 로그 추가

      if (response.data.image) {
        const imageSrc = `data:image/png;base64,${response.data.image}`;
        console.log('Generated Image:', imageSrc);  // 콘솔 로그 추가
        setImage(imageSrc);
      }
      setMessage(response.data.message);
      console.log('Message:', response.data.message);  // 콘솔 로그 추가
    } catch (error) {
      setMessage('Error occurred while processing the image.');
      console.error('Error:', error);  // 콘솔 로그 추가
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
