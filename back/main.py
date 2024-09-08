from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler
import openai
import base64
import cv2
import mediapipe as mp
import io

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 키 설정
openai.api_key = ""  # 여기에 OpenAI API 키 입력

# 모델 로드 함수
def load_models(model_id: str, controlnet_ids: dict):
    try:
        print(f"Loading models: {model_id} and {list(controlnet_ids.values())}")
        sd_model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        
        controlnet_models = {}
        for key, controlnet_id in controlnet_ids.items():
            try:
                controlnet_model = ControlNetModel.from_pretrained(controlnet_id)
                controlnet_models[key] = controlnet_model
            except Exception as e:
                print(f"Error loading ControlNet model {controlnet_id}: {e}")
        
        sd_model.controlnet = controlnet_models
        sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config)
        sd_model.to("cuda" if torch.cuda.is_available() else "cpu")

        print("Models loaded successfully")
        return sd_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise RuntimeError(f"Error loading models: {e}")

# 사용자 정의 체크포인트 및 ControlNet 경로 설정
model_id = "digiplay/Cetus-Mix-Codaedition_diffusers"
controlnet_ids = {
    "openpose": "lllyasviel/sd-controlnet-openpose",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "normalmap": "tori29umai/control_v11p_sd21_normalmap_diffusers",
    "depthmap": "SargeZT/controlnet-sd-xl-1.0-depth-zeed"
}
sd_model = load_models(model_id, controlnet_ids)

# 스케치 이미지의 디테일을 개선하는 함수 (업그레이드된 버전)
def enhance_sketch_image(image: np.array) -> np.array:
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 선명한 스케치 라인 만들기 (Canny Edge Detection 사용)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # 작은 노이즈 제거 및 선 두께 개선
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)  # 선을 두껍게 만듦
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)  # 선을 적절히 얇게 복구

    # 선의 흐름을 부드럽게 연결하여 자연스럽게 만듦
    blurred_edges = cv2.GaussianBlur(edges_eroded, (3, 3), 0)

    # 기존 선을 기반으로 끊긴 부분을 복구
    contours, _ = cv2.findContours(blurred_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sketch_contours = np.zeros_like(blurred_edges)
    cv2.drawContours(sketch_contours, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 스케치 라인을 부드럽게 하면서 선명도를 유지하는 필터 적용
    sketch_refined = cv2.adaptiveThreshold(sketch_contours, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 해상도 향상 처리 (super_image 등의 라이브러리 활용 가능)
    # enhanced_image = enhance_image_resolution(sketch_refined)

    return cv2.cvtColor(sketch_refined, cv2.COLOR_GRAY2RGB)  # 결과 이미지를 RGB로 변환하여 반환

# OpenPose 전처리기
def preprocess_openpose(image: np.array) -> np.array:
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image_rgb)
    
    # 스케치 효과 적용
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    
    # 스케치 필터링 (선 정리)
    sketch_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    sketch_image = cv2.GaussianBlur(sketch_image, (5, 5), 0)  # 부드러운 블러
    sketch_image = cv2.adaptiveThreshold(sketch_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 선의 형태 보강
    kernel = np.ones((3,3), np.uint8)
    sketch_image = cv2.dilate(sketch_image, kernel, iterations=1)
    sketch_image = cv2.erode(sketch_image, kernel, iterations=1)

    return cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2RGB)  # 이미지 채널을 다시 RGB로 변환

# 포즈 추출 함수
def extract_pose(image: np.array) -> dict:
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image_rgb)
    
    pose_landmarks = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose_landmarks = {
            'left_shoulder': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y),
            'left_elbow': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y),
            'left_wrist': (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y),
            'right_shoulder': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            'right_elbow': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y),
            'right_wrist': (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y)
        }
    return pose_landmarks

# 포즈 유사성 계산 함수
def calculate_angle(p1, p2, p3):
    vec1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vec2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
    return np.degrees(angle)

def compare_pose(original_pose: dict, generated_pose: dict) -> float:
    total_distance = 0.0
    num_comparisons = 0

    for side in ['left', 'right']:
        if original_pose.get(f'{side}_shoulder') is None or generated_pose.get(f'{side}_shoulder') is None:
            continue

        shoulder = (original_pose.get(f'{side}_shoulder'), generated_pose.get(f'{side}_shoulder'))
        elbow = (original_pose.get(f'{side}_elbow'), generated_pose.get(f'{side}_elbow'))
        wrist = (original_pose.get(f'{side}_wrist'), generated_pose.get(f'{side}_wrist'))

        if None not in shoulder + elbow + wrist:
            shoulder_elbow_angle_original = calculate_angle(shoulder[0], elbow[0], wrist[0])
            shoulder_elbow_angle_generated = calculate_angle(shoulder[1], elbow[1], wrist[1])

            total_distance += abs(shoulder_elbow_angle_original - shoulder_elbow_angle_generated)
            num_comparisons += 1

    if num_comparisons == 0:
        return 0.0  # 유사도 점수를 0으로 설정하여 무한대 대신 사용할 수 있습니다.

    average_distance = total_distance / num_comparisons
    return average_distance

# OpenAI를 사용하여 동작 설명 생성
def generate_openai_action_description(image: Image.Image) -> str:
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        response = openai.Image.create(
            file=img_base64,
            model="dall-e",
            prompt="Describe the action or movement happening in this image in one sentence."
        )
        action_description = response['choices'][0]['text'].strip()
        return action_description
    except Exception as e:
        print(f"Error generating action description: {e}")
        return "Unable to describe the action."

# 스케치 모드에서 흑백 선화 생성 (하얀 배경에 검은 선)
def generate_line_drawing_with_prompt(base_image: np.array, prompt: str) -> np.array:
    # 1. 프롬프트를 적용하여 새로운 디자인을 추가한 이미지를 생성
    generated_design = sd_model(prompt=prompt, negative_prompt="lowres, bad anatomy, text, worst quality", image=Image.fromarray(base_image), strength=0.8).images[0]
    
    # 2. 생성된 이미지를 흑백으로 변환
    generated_design_np = np.array(generated_design.convert("L"))  # 흑백 변환
    
    # 3. 하얀 배경에 검은 선만 남도록 Canny Edge Detection 적용
    edges = cv2.Canny(generated_design_np, threshold1=50, threshold2=150)
    
    # 4. 하얀 배경으로 변환 (모든 픽셀을 255로 설정하고, 검은 선만 남김)
    line_drawing = np.ones_like(edges) * 255  # 모든 배경을 하얗게 설정
    line_drawing[edges == 255] = 0  # 선을 검은색으로 설정

    # 5. 결과물을 RGB로 변환 (흑백 선화이지만 3채널로 맞추기 위해)
    line_drawing_rgb = cv2.cvtColor(line_drawing, cv2.COLOR_GRAY2RGB)
    
    return line_drawing_rgb

# 이미지 생성 엔드포인트
@app.post("/generate/")
async def generate_image(
    file: UploadFile = File(...), 
    prompt: str = Form(...), 
    similarity_threshold: float = Form(10.0),
    sketch_mode: bool = Form(False)  # 스케치 모드 매개변수 추가
):
    try:
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad feet, bad legs, extra legs, extra limb, extra arm, extra hands, twisted fingers, cut fingers, weird fingers, weird hands, twisted hands, extra fingers, bad fingers,"
        
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        if sketch_mode:
            # 스케치 모드일 때는 스케치 처리 및 품질 개선 적용
            processed_image_np = preprocess_openpose(image_np)  # 기존 전처리
            enhanced_image_np = enhance_sketch_image(processed_image_np)  # 고품질 스케치로 변환
            
            # 프롬프트에 따라 디테일을 추가한 흑백 선화 생성 (하얀 배경, 검은 선)
            final_line_drawing = generate_line_drawing_with_prompt(enhanced_image_np, prompt)
            
            # 최종 선화 이미지를 저장하여 반환
            output = io.BytesIO()
            Image.fromarray(final_line_drawing).save(output, format='PNG')
            output.seek(0)
            return JSONResponse(content={"message": "스케치 및 디테일 선화 생성 성공", "image": base64.b64encode(output.getvalue()).decode('utf-8')})
        
        # 스케치 모드가 아닌 경우 기존 로직 수행
        original_pose = extract_pose(image_np)
        action_description = generate_openai_action_description(image)
        situation_description = "Some situation"
        final_prompt = f"{prompt} {action_description} {situation_description}"

        similarity_score = float('inf')
        generated_image = None

        while similarity_score == float('inf'):
            generated_image = sd_model(prompt=final_prompt, negative_prompt=negative_prompt, num_inference_steps=120, image=image, strength=0.5).images[0]
            
            if generated_image is not None:
                generated_image_np = np.array(generated_image)
                generated_pose = extract_pose(generated_image_np)
                similarity_score = compare_pose(original_pose, generated_pose)
                
                if similarity_score > similarity_threshold:
                    similarity_score = float('inf')
                else:
                    break

        if generated_image is not None:
            output = io.BytesIO()
            generated_image.save(output, format='PNG')
            output.seek(0)
            return JSONResponse(content={"message": "이미지 생성 성공", "image": base64.b64encode(output.getvalue()).decode('utf-8')})

        return JSONResponse(content={"message": "이미지 생성 실패"})

    except Exception as e:
        return JSONResponse(content={"message": f"이미지 처리 중 오류 발생: {e}"})
