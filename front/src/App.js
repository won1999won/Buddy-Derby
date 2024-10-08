import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [file1, setFile1] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [message, setMessage] = useState('');
  
  const [generatedImageT2I, setGeneratedImageT2I] = useState(null);
  const [messageT2I, setMessageT2I] = useState('');

  // LoRA 파일 목록 상태 추가
  const [loraFiles, setLoraFiles] = useState([]);
  const [selectedLoraFile, setSelectedLoraFile] = useState('');
  
  const [applyLoraFlag, setApplyLoraFlag] = useState(false);
  const [minFaceSize, setMinFaceSize] = useState(100);
  const [controlnetStrength, setControlnetStrength] = useState(0.5);

  // 컴포넌트 마운트 시 LoRA 파일 목록 가져오기
  useEffect(() => {
    const fetchLoraFiles = async () => {
      try {
        const response = await axios.get('http://localhost:5000/list_lora_files/');
        setLoraFiles(response.data.lora_files);
        if (response.data.lora_files.length > 0) {
          setSelectedLoraFile(response.data.lora_files[0]); // 기본 선택값 설정
        }
      } catch (error) {
        console.error('LoRA 파일 목록 가져오기 실패:', error);
      }
    };
    fetchLoraFiles();
  }, []);

  // i2i 요청 처리
  const handleGenerateSubmit = async (event) => {
    event.preventDefault();
    setMessage('');
    setGeneratedImage(null);

    if (!file1) {
      setMessage('생성할 이미지를 선택해주세요.');
      return;
    }

    if (!prompt.trim()) {
      setMessage('프롬프트를 입력해주세요.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file1);
    formData.append('prompt', prompt);
    formData.append('controlnet_strength', controlnetStrength);
    formData.append('min_face_size', minFaceSize);
    formData.append('apply_lora_flag', applyLoraFlag);

    if (applyLoraFlag && selectedLoraFile) {
      formData.append('lora_file_name', selectedLoraFile); // 파일 이름 전송
    }

    // 폼 데이터 내용을 콘솔에 출력
    console.log('전송할 데이터 (i2i):');
    formData.forEach((value, key) => {
      if (key === 'file' || key === 'lora_file_name') {
        console.log(`${key}:`, value); // 파일 이름 또는 파일 객체 출력
      } else {
        console.log(`${key}:`, value); // 다른 값은 그대로 출력
      }
    });

    try {
      const response = await axios.post('http://localhost:5000/generate/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.image) {
        const imageSrc = `data:image/png;base64,${response.data.image}`;
        setGeneratedImage(imageSrc);
      }
      setMessage(response.data.message);
    } catch (error) {
      console.error('오류 발생 (i2i):', error);
      if (error.response && error.response.data && error.response.data.message) {
        setMessage(`오류: ${error.response.data.message}`);
      } else {
        setMessage('이미지 처리 중 예상치 못한 오류가 발생했습니다.');
      }
    }
  };

  // t2i 요청 처리
  const handleGenerateT2ISubmit = async (event) => {
    event.preventDefault();
    setMessageT2I('');
    setGeneratedImageT2I(null);

    if (!prompt.trim()) {
      setMessageT2I('프롬프트를 입력해주세요.');
      return;
    }

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('apply_lora_flag', applyLoraFlag);

    if (applyLoraFlag && selectedLoraFile) {
      formData.append('lora_file_name', selectedLoraFile); // 파일 이름 전송
    }

    // 폼 데이터 내용을 콘솔에 출력
    console.log('전송할 데이터 (T2I):');
    formData.forEach((value, key) => {
      if (key === 'lora_file_name') {
        console.log(`${key}:`, value); // 파일 이름 출력
      } else {
        console.log(`${key}:`, value); // 다른 값은 그대로 출력
      }
    });

    try {
      const response = await axios.post('http://localhost:5000/generate_t2i/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.image) {
        const imageSrc = `data:image/png;base64,${response.data.image}`;
        setGeneratedImageT2I(imageSrc);
      }
      setMessageT2I(response.data.message);
    } catch (error) {
      console.error('오류 발생 (T2I):', error);
      if (error.response && error.response.data && error.response.data.message) {
        setMessageT2I(`오류: ${error.response.data.message}`);
      } else {
        setMessageT2I('이미지 처리 중 예상치 못한 오류가 발생했습니다.');
      }
    }
  };

  return (
    <div className="App" style={{ padding: '20px' }}>
      <h1>이미지 생성 및 스케치 처리 (i2i)</h1>
      <form onSubmit={handleGenerateSubmit}>
        <div>
          <label>
            생성용 첫 번째 이미지:
            <input type="file" accept="image/*" onChange={(e) => setFile1(e.target.files[0])} required />
          </label>
        </div>
        <div>
          <label>
            프롬프트:
            <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} required />
          </label>
        </div>
        <div>
          {/* <label>
            LoRA 모델 적용:
            <input type="checkbox" checked={applyLoraFlag} onChange={(e) => setApplyLoraFlag(e.target.checked)} />
          </label> */}
        </div>
        {applyLoraFlag && (
          <div>
            <label>
              LoRA 파일 선택:
              <select value={selectedLoraFile} onChange={(e) => setSelectedLoraFile(e.target.value)}>
                {loraFiles.map((file, index) => (
                  <option key={index} value={file}>
                    {file}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
        <div>
          <label>
            최소 얼굴 크기:
            <input type="number" value={minFaceSize} onChange={(e) => setMinFaceSize(Number(e.target.value))} />
          </label>
        </div>
        <div>
          <label>
            ControlNet 강도:
            <input type="range" min="0" max="1" step="0.1" value={controlnetStrength} onChange={(e) => setControlnetStrength(Number(e.target.value))} />
          </label>
        </div>
        <button type="submit">이미지 생성 (i2i)</button>
      </form>

      {message && <p>{message}</p>}

      {generatedImage && (
        <div>
          <img src={generatedImage} alt="Generated" style={{ maxWidth: '50%' }} />
          <a href={generatedImage} download="generated_image.png">이미지 다운로드</a>
        </div>
      )}

      <h1>텍스트-투-이미지 생성 (t2i)</h1>
      <form onSubmit={handleGenerateT2ISubmit}>
        <div>
          <label>
            프롬프트:
            <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} required />
          </label>
        </div>
        <div>
          {/* <label>
            LoRA 모델 적용:
            <input type="checkbox" checked={applyLoraFlag} onChange={(e) => setApplyLoraFlag(e.target.checked)} />
          </label> */}
        </div>
        {applyLoraFlag && (
          <div>
            <label>
              LoRA 파일 선택:
              <select value={selectedLoraFile} onChange={(e) => setSelectedLoraFile(e.target.value)}>
                {loraFiles.map((file, index) => (
                  <option key={index} value={file}>
                    {file}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
        <button type="submit">이미지 생성 (t2i)</button>
      </form>

      {messageT2I && <p>{messageT2I}</p>}

      {generatedImageT2I && (
        <div>
          <img src={generatedImageT2I} alt="Generated T2I" style={{ maxWidth: '100%' }} />
          <a href={generatedImageT2I} download="generated_image_t2i.png">이미지 다운로드</a>
        </div>
      )}
    </div>
  );
};

// 로깅 객체를 사용하기 위해 window에 추가 (개발용)
const logger = {
  debug: (...args) => console.debug(...args),
  info: (...args) => console.info(...args),
  warn: (...args) => console.warn(...args),
  error: (...args) => console.error(...args),
};

export default App;
