from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
import io
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler
import openai
import base64
import cv2
import mediapipe as mp
import random

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
openai.api_key = ""

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
        shoulder = (original_pose.get(f'{side}_shoulder'), generated_pose.get(f'{side}_shoulder'))
        elbow = (original_pose.get(f'{side}_elbow'), generated_pose.get(f'{side}_elbow'))
        wrist = (original_pose.get(f'{side}_wrist'), generated_pose.get(f'{side}_wrist'))
        
        if None not in shoulder + elbow + wrist:
            shoulder_elbow_angle_original = calculate_angle(shoulder[0], elbow[0], wrist[0])
            shoulder_elbow_angle_generated = calculate_angle(shoulder[1], elbow[1], wrist[1])
            
            total_distance += abs(shoulder_elbow_angle_original - shoulder_elbow_angle_generated)
            num_comparisons += 1

    average_distance = total_distance / num_comparisons if num_comparisons > 0 else float('inf')
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

# OpenAI를 사용하여 상황 묘사 생성
def generate_openai_situation_description(image: Image.Image) -> str:
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        response = openai.Image.create(
            file=img_base64,
            model="dall-e",
            prompt="Describe the scene or context of this image in one sentence."
        )
        situation_description = response['choices'][0]['text'].strip()
        return situation_description
    except Exception as e:
        print(f"Error generating situation description: {e}")
        return "Unable to describe the situation."

# OpenAI를 사용하여 이미지 평가
def generate_openai_feedback(image_description: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an art critic who provides constructive feedback on digital art."},
                {"role": "user", "content": f"Evaluate the quality of the image described below and suggest improvements.\n\nImage description: {image_description}"}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error generating OpenAI feedback: {e}")
        return "Feedback generation failed."

# 프롬프트를 간단하게 하기 위한 함수
def truncate_prompt(prompt: str, max_length: int = 100) -> str:
    if len(prompt) > max_length:
        return prompt[:max_length] + "..."
    return prompt

# 이미지 조정 함수
def adjust_image_quality(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # 이미지 선명도 증가
    return image

# 얼굴 복원 함수
def restore_faces(image: Image.Image) -> Image.Image:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return Image.fromarray(image_cv)

# 이미지 전처리 함수
def apply_openpose_preprocessor(image: np.array) -> np.array:
    target_size = (256, 256)
    return cv2.resize(image, target_size)

# 얼굴, 손, 배경 업스케일링 함수
def upscale_face_hands_background(image: Image.Image) -> Image.Image:
    image_cv = np.array(image)
    scale_factor = 2
    height, width = image_cv.shape[:2]
    new_size = (width * scale_factor, height * scale_factor)
    upscaled_image = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(upscaled_image)

# 몸체 회전 각도 계산 함수
def calculate_body_rotation(pose_landmarks: dict) -> tuple:
    chest_angle = random.uniform(0, 360)  # 예시 값
    waist_angle = random.uniform(0, 360)  # 예시 값
    return chest_angle, waist_angle

# FastAPI 엔드포인트 정의
@app.post("/generate/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad feet, bad legs, extra legs, extra limb, extra arm, extra hands, twisted fingers, cut fingers, weird fingers, weird hands, twisted hands, extra fingers, bad fingers,"

        image = Image.open(file.file).convert("RGB")
        image = adjust_image_quality(image)
        image = restore_faces(image)
        
        image_np = np.array(image)
        pose_landmarks = extract_pose(image_np)
        
        if pose_landmarks:
            chest_angle, waist_angle = calculate_body_rotation(pose_landmarks)
            print(f"Original chest angle: {chest_angle}, waist angle: {waist_angle}")

            image_np = apply_openpose_preprocessor(image_np)
            image = Image.fromarray(image_np)

            image = upscale_face_hands_background(image)

            # 입력된 이미지로 동작 및 상황 설명 생성
            action_description = generate_openai_action_description(image)
            situation_description = generate_openai_situation_description(image)

            # 입력 프롬프트와 동작 설명 및 상황 설명 결합
            final_prompt = f"{prompt} {action_description} {situation_description}"

            # 포즈를 고려한 프롬프트 생성
            pose_prompt = f"Ensure the pose matches the following: {action_description}"
            final_prompt_with_pose = f"{pose_prompt} {final_prompt}"

            output_image = sd_model(
                prompt=final_prompt_with_pose,
                image=image,
                strength=0.75,
                negative_prompt=negative_prompt,
                controlnet=sd_model.controlnet,
                guidance_scale=15.0,
                num_inference_steps=350,
                controlnet_conditioning_scale=6.0
            ).images[0]

            generated_pose_image = np.array(output_image)
            generated_pose_landmarks = extract_pose(generated_pose_image)
            
            if generated_pose_landmarks:
                generated_chest_angle, generated_waist_angle = calculate_body_rotation(generated_pose_landmarks)
                pose_similarity = compare_pose(pose_landmarks, generated_pose_landmarks)
            else:
                generated_chest_angle, generated_waist_angle = None, None
                pose_similarity = float('inf')

            # 포즈 유사성이 좋지 않으면 이미지 재생성
            max_attempts = 3
            attempt = 0
            while pose_similarity > 10.0 and attempt < max_attempts:
                attempt += 1
                output_image = sd_model(
                    prompt=final_prompt_with_pose,
                    image=image,
                    strength=0.75,
                    negative_prompt=negative_prompt,
                    controlnet=sd_model.controlnet,
                    guidance_scale=15.0,
                    num_inference_steps=350,
                    controlnet_conditioning_scale=6.0
                ).images[0]
                
                generated_pose_image = np.array(output_image)
                generated_pose_landmarks = extract_pose(generated_pose_image)
                
                if generated_pose_landmarks:
                    generated_chest_angle, generated_waist_angle = calculate_body_rotation(generated_pose_landmarks)
                    pose_similarity = compare_pose(pose_landmarks, generated_pose_landmarks)
                else:
                    pose_similarity = float('inf')

            feedback = generate_openai_feedback(action_description)

            output_image_path = "output_image.png"
            output_image.save(output_image_path)

            with open(output_image_path, "rb") as image_file:
                output_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            pose_image_base64 = base64.b64encode(generated_pose_image).decode('utf-8') if generated_pose_image is not None else None

            return JSONResponse(content={
                "description": action_description,
                "situation": situation_description,
                "feedback": feedback,
                "image": output_image_base64,
                "pose_image": pose_image_base64,
                "chest_angle": generated_chest_angle,
                "waist_angle": generated_waist_angle
            })
    except Exception as e:
        print(f"Error during image processing: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
