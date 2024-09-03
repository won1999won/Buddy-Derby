import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [poseImage, setPoseImage] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  const [manualPrompt, setManualPrompt] = useState('');
  const [useManualPrompt, setUseManualPrompt] = useState(false);

  const handleFileChange = (event) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      alert('이미지를 선택해 주세요.');
      return;
    }

    setLoading(true);
    setProcessedImage(null);
    setPoseImage(null);
    setFeedback('');

    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('prompt', useManualPrompt ? manualPrompt : ""); // 프롬프트 처리

    try {
      const response = await axios.post('http://localhost:5000/generate/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      const { image, pose_image, feedback } = response.data;
      setProcessedImage(`data:image/png;base64,${image}`);
      setPoseImage(pose_image ? `data:image/png;base64,${pose_image}` : null);
      setFeedback(feedback);
    } catch (error) {
      console.error('이미지 업로드 오류:', error);
      alert('이미지 업로드 및 처리에 실패했습니다. 콘솔에서 자세한 내용을 확인하세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Stable Diffusion 이미지 처리</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      <div>
        <label>
          <input
            type="checkbox"
            checked={useManualPrompt}
            onChange={() => setUseManualPrompt(!useManualPrompt)}
          />
          수동 프롬프트 사용
        </label>
      </div>

      {useManualPrompt && (
        <textarea
          value={manualPrompt}
          onChange={(e) => setManualPrompt(e.target.value)}
          placeholder="여기에 사용자 정의 프롬프트를 입력하세요"
          style={{ width: '100%', height: '100px' }}
        />
      )}
      
      <button onClick={handleUpload}>업로드 및 처리</button>
      
      {loading && <p>이미지를 처리 중입니다. 잠시만 기다려 주세요...</p>}
      
      {poseImage && (
        <div>
          <h2>포즈 이미지:</h2>
          <img src={poseImage} alt="Pose" style={{ maxWidth: '100%' }} />
        </div>
      )}

      {processedImage && (
        <div>
          <h2>처리된 이미지:</h2>
          <img src={processedImage} alt="Processed" style={{ maxWidth: '100%' }} />
        </div>
      )}
      
      {feedback && (
        <div>
          <h2>피드백:</h2>
          <p>{feedback}</p>
        </div>
      )}
    </div>
  );
}

export default App;
