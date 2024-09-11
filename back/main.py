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
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

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
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)
    blurred_edges = cv2.GaussianBlur(edges_eroded, (3, 3), 0)
    contours, _ = cv2.findContours(blurred_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sketch_contours = np.zeros_like(blurred_edges)
    cv2.drawContours(sketch_contours, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    sketch_refined = cv2.adaptiveThreshold(sketch_contours, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(sketch_refined, cv2.COLOR_GRAY2RGB)

# OpenPose 전처리기
def preprocess_openpose(image: np.array) -> np.array:
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image_rgb)
    
    pose_image = np.zeros_like(image)
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(pose_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    
    return pose_image

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
        return 0.0

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
def generate_line_drawing_with_pose(base_image: np.array, prompt: str, pose_image: np.array, negative_prompt: str, controlnet_strength: float = 0.5) -> np.array:
    pose_pil = Image.fromarray(pose_image)
    generated_design = sd_model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(base_image),
        controlnet_conditioning_image=pose_pil,
        strength=controlnet_strength
    ).images[0]

    generated_design_np = np.array(generated_design.convert("L"))
    edges = cv2.Canny(generated_design_np, threshold1=50, threshold2=150)
    line_drawing = np.ones_like(edges) * 255
    line_drawing[edges == 255] = 0
    line_drawing_rgb = cv2.cvtColor(line_drawing, cv2.COLOR_GRAY2RGB)

    return line_drawing_rgb

# 얼굴, 손, 발을 탐지하는 함수
def detect_face_hands_feet(image: np.array) -> dict:
    results = {}

    # 얼굴 탐지
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    face_results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if face_results.multi_face_landmarks:
        face_landmarks = [face_landmark.landmark for face_landmark in face_results.multi_face_landmarks]
        results['faces'] = face_landmarks

    # 손 탐지
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
    hand_results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if hand_results.multi_hand_landmarks:
        hand_landmarks = [hand_landmark.landmark for hand_landmark in hand_results.multi_hand_landmarks]
        results['hands'] = hand_landmarks

    return results

# 얼굴, 손 영역 보정
def refine_face_hands_feet(image: np.array, landmarks: dict) -> np.array:
    refined_image = image.copy()

    if 'faces' in landmarks:
        for face_landmark in landmarks['faces']:
            face_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmark]
            face_bbox = cv2.boundingRect(np.array(face_coords))
            face_roi = refined_image[face_bbox[1]:face_bbox[1] + face_bbox[3], face_bbox[0]:face_bbox[0] + face_bbox[2]]

    if 'hands' in landmarks:
        for hand_landmark in landmarks['hands']:
            hand_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in hand_landmark]
            hand_bbox = cv2.boundingRect(np.array(hand_coords))
            hand_roi = refined_image[hand_bbox[1]:hand_bbox[1] + hand_bbox[3], hand_bbox[0]:hand_bbox[0] + hand_bbox[2]]

    return refined_image

# 옷 영역 탐지 및 보정
def detect_clothes(image: np.array) -> np.array:
    print("Detecting clothes")
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).eval()

    image_pil = Image.fromarray(image)
    preprocess = weights.transforms()
    input_tensor = preprocess(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    clothes_mask = (output_predictions == 17).astype(np.uint8) * 255
    if clothes_mask.shape != image.shape[:2]:
        clothes_mask = cv2.resize(clothes_mask, (image.shape[1], image.shape[0]))

    return clothes_mask

def refine_clothes(image: np.array, clothes_mask: np.array) -> np.array:
    print("Refining clothes")
    refined_image = image.copy()

    clothes_region = cv2.bitwise_and(image, image, mask=clothes_mask)
    clothes_region_enhanced = cv2.detailEnhance(clothes_region, sigma_s=10, sigma_r=0.15)
    refined_image[clothes_mask == 255] = clothes_region_enhanced[clothes_mask == 255]

    return refined_image

# Stable Diffusion 업스케일러 로드
def load_sd_upscaler():
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return sd_pipeline

sd_upscaler = load_sd_upscaler()

# 특정 영역 업스케일
def upscale_with_stable_diffusion(image: np.array, region: tuple) -> np.array:
    print("Upscaling with Stable Diffusion")
    x, y, w, h = region
    region_image = image[y:y+h, x:x+w]
    region_pil = Image.fromarray(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        upscaled_region = sd_upscaler(prompt="", image=region_pil, strength=1).images[0]

    upscaled_region_np = np.array(upscaled_region)
    upscaled_region_bgr = cv2.cvtColor(upscaled_region_np, cv2.COLOR_RGB2BGR)
    image[y:y+h, x:x+w] = upscaled_region_bgr

    return image

# 얼굴, 손, 무릎 영역 업스케일
def apply_upscale_on_regions(image: np.array, landmarks: dict, min_face_size: int = 100) -> np.array:
    refined_image = image.copy()
    refined_image = upscale_small_faces(refined_image, landmarks, min_face_size)

    if 'hands' in landmarks:
        for hand_landmark in landmarks['hands']:
            hand_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in hand_landmark]
            hand_bbox = cv2.boundingRect(np.array(hand_coords))
            refined_image = upscale_with_stable_diffusion(refined_image, hand_bbox)

    return refined_image

# 작은 얼굴 업스케일
def upscale_small_faces(image: np.array, landmarks: dict, min_face_size: int = 100) -> np.array:
    print("Upscaling small faces")
    refined_image = image.copy()

    if 'faces' in landmarks:
        for face_landmark in landmarks['faces']:
            face_size = detect_face_size(image, [face_landmark])
            if face_size < min_face_size:
                face_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmark]
                face_bbox = cv2.boundingRect(np.array(face_coords))
                refined_image = upscale_with_stable_diffusion(refined_image, face_bbox)

    return refined_image

# Mediapipe 얼굴 크기 감지
def detect_face_size(image: np.array, face_landmarks) -> float:
    if not face_landmarks:
        return 0
    face_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks[0]]
    face_bbox = cv2.boundingRect(np.array(face_coords))
    return face_bbox[2] * face_bbox[3]

# 이미지 생성 엔드포인트
@app.post("/generate/")
async def generate_image(
    file: UploadFile = File(...), 
    prompt: str = Form(...), 
    similarity_threshold: float = Form(10.0),
    sketch_mode: bool = Form(False),
    controlnet_strength: float = Form(0.5),
    min_face_size: int = Form(100)
):
    try:
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad feet, bad legs, extra legs, extra limb, extra arm, extra hands, twisted fingers, cut fingers, weird fingers, weird hands, twisted hands, extra fingers, bad fingers,"
        
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        if sketch_mode:
            pose_image = preprocess_openpose(image_np)
            final_line_drawing = generate_line_drawing_with_pose(image_np, prompt, pose_image, negative_prompt, controlnet_strength)

            clothes_mask = detect_clothes(image_np)
            final_line_drawing = refine_clothes(final_line_drawing, clothes_mask)
            
            landmarks = detect_face_hands_feet(image_np)
            final_line_drawing = apply_upscale_on_regions(final_line_drawing, landmarks, min_face_size)
            
            output = io.BytesIO()
            Image.fromarray(final_line_drawing).save(output, format='PNG')
            output.seek(0)
            return JSONResponse(content={"message": "스케치 및 옷 보정 성공", "image": base64.b64encode(output.getvalue()).decode('utf-8')})
        
        original_pose = extract_pose(image_np)
        action_description = generate_openai_action_description(image)
        situation_description = "Some situation"
        final_prompt = f"{prompt} {action_description} {situation_description}"

        similarity_score = float('inf')
        generated_image = None

        while similarity_score == float('inf'):
            generated_image = sd_model(
                prompt=final_prompt, 
                negative_prompt=negative_prompt,  
                num_inference_steps=120, 
                image=image, 
                strength=0.5
            ).images[0]
            
            if generated_image is not None:
                generated_image_np = np.array(generated_image)
                generated_pose = extract_pose(generated_image_np)
                similarity_score = compare_pose(original_pose, generated_pose)
                
                if similarity_score > similarity_threshold:
                    similarity_score = float('inf')
                else:
                    break

        if generated_image is not None:
            clothes_mask = detect_clothes(image_np)
            generated_image_np = refine_clothes(generated_image_np, clothes_mask)
            
            landmarks = detect_face_hands_feet(image_np)
            generated_image_np = apply_upscale_on_regions(generated_image_np, landmarks, min_face_size)
            
            output = io.BytesIO()
            Image.fromarray(generated_image_np).save(output, format='PNG')
            output.seek(0)
            return JSONResponse(content={"message": "이미지 생성 성공", "image": base64.b64encode(output.getvalue()).decode('utf-8')})

        return JSONResponse(content={"message": "이미지 생성 실패"})

    except Exception as e:
        return JSONResponse(content={"message": f"이미지 처리 중 오류 발생: {e}"})
