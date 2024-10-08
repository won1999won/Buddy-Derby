from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
import os
import logging
import io
import base64
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from typing import Optional
import traceback  # 추가된 부분

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)  # 디버그 레벨로 설정
logger = logging.getLogger(__name__)

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

# 기본 LoRA 디렉토리 설정
DEFAULT_LORA_DIR = r"C:\Users\user\Desktop\Buddy Derby\back\LORA"  # 경로 확인 및 수정

# 모델 로드 함수
def load_models(model_id: str, controlnet_ids: dict):
    try:
        logger.debug(f"Loading models: {model_id} and ControlNet models: {list(controlnet_ids.values())}")
        controlnet_models = {}
        for key, controlnet_id in controlnet_ids.items():
            try:
                controlnet_model = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
                controlnet_models[key] = controlnet_model
                logger.debug(f"Loaded ControlNet model: {controlnet_id}")
            except Exception as e:
                logger.error(f"Error loading ControlNet model {controlnet_id}: {e}", exc_info=True)
        
        sd_img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=list(controlnet_models.values()),
            torch_dtype=torch.float16
        )
        sd_img2img_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_img2img_model.scheduler.config)
        sd_img2img_model = sd_img2img_model.to("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Stable Diffusion Img2Img Pipeline loaded successfully")
        
        # 불필요한 UNet 레이어 필터링 로그 제거

        sd_txt2img_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        sd_txt2img_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_txt2img_model.scheduler.config)
        sd_txt2img_model = sd_txt2img_model.to("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Stable Diffusion Txt2Img Pipeline loaded successfully")

        # 불필요한 UNet 레이어 필터링 로그 제거
        
        return sd_img2img_model, sd_txt2img_model, controlnet_models
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        raise RuntimeError(f"Error loading models: {e}")


model_id = "digiplay/Cetus-Mix-Codaedition_diffusers"
controlnet_ids = {
    "openpose": "lllyasviel/sd-controlnet-openpose",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "normalmap": "tori29umai/control_v11p_sd21_normalmap_diffusers",
    "depthmap": "SargeZT/controlnet-sd-xl-1.0-depth-zeed",
    "tile": "lllyasviel/control_v11f1e_sd15_tile"
}

sd_img2img_model, sd_txt2img_model, controlnet_models = load_models(model_id, controlnet_ids)

# LoRA 모델 로드 및 적용 함수
def load_and_apply_lora(pipeline, lora_file_path):
    try:
        logger.debug(f"Loading LoRA model from file: {lora_file_path}")

        if not os.path.isfile(lora_file_path):
            logger.error(f"LoRA 모델 파일이 아닙니다: {lora_file_path}")
            return False  # LoRA 적용을 건너뜁니다.

        # 파이프라인의 UNet 레이어 정보 로깅 (디버깅 용도)
        logger.debug("Pipeline UNet layers:")
        for name, param in pipeline.unet.named_parameters():
            logger.debug(f"{name}: {param.shape}")

        # Diffusers에서 LoRA 모델 가중치 로드
        logger.debug("Loading LoRA model using load_attn_procs on UNet...")

        # LoRA 모델 로드 방법 적용 (medium, tistory에서 제시된 방식 사용)
        pipeline.unet.load_attn_procs(lora_file_path)
        logger.debug(f"Successfully applied LoRA model from: {lora_file_path}")

        return True
    except Exception as e:
        # 전체 스택 트레이스를 로깅하고 LoRA 적용을 건너뜁니다.
        logger.error(f"Error loading and applying LoRA model: {e}")
        logger.error(traceback.format_exc())
        return False

# 얼굴 랜드마크 탐지 함수
def detect_face_landmarks(image: np.array) -> dict:
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    face_landmarks_dict = {}

    logger.debug("Detecting face landmarks...")
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_coords = [
                (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                for landmark in face_landmarks.landmark
            ]
            face_bbox = cv2.boundingRect(np.array(face_coords))
            face_landmarks_dict['face_bbox'] = face_bbox
            face_landmarks_dict['face_coords'] = face_coords
            logger.debug(f"Detected face_bbox: {face_bbox}")
    else:
        logger.debug("No face landmarks detected.")

    return face_landmarks_dict

# 얼굴, 손, 발 탐지 함수
def detect_face_hands_feet(image: np.array) -> dict:
    results = {}
    face_landmarks_dict = detect_face_landmarks(image)
    if face_landmarks_dict:
        results['face_bbox'] = face_landmarks_dict['face_bbox']
        results['faces'] = face_landmarks_dict['face_coords']
        logger.debug(f"Face landmarks detected: {face_landmarks_dict['face_bbox']}")

    mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
    logger.debug("Detecting hands...")
    hand_results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if hand_results.multi_hand_landmarks:
        hand_landmarks = [
            hand_landmark.landmark for hand_landmark in hand_results.multi_hand_landmarks
        ]
        results['hands'] = hand_landmarks
        logger.debug(f"Detected hands: {len(hand_landmarks)}")
    else:
        logger.debug("No hands detected.")

    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    logger.debug("Detecting pose...")
    pose_results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if pose_results.pose_landmarks:
        feet_landmarks = {
            'left_foot_index': (
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
            ),
            'right_foot_index': (
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
            ),
        }
        results['feet'] = feet_landmarks
        logger.debug(f"Detected feet landmarks: {feet_landmarks}")
    else:
        logger.debug("No pose landmarks detected.")

    return results

# 옷 영역 탐지 함수
def detect_clothes(image: np.array) -> np.array:
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).eval()
    logger.debug("Detecting clothes...")
    
    image_pil = Image.fromarray(image)
    preprocess = weights.transforms()
    input_tensor = preprocess(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # 클래스 ID 15은 'person' 클래스입니다. (DeepLabV3의 COCO 데이터셋 기준)
    clothes_mask = (output_predictions == 15).astype(np.uint8) * 255
    if clothes_mask.shape != image.shape[:2]:
        clothes_mask = cv2.resize(clothes_mask, (image.shape[1], image.shape[0]))
        logger.debug("Resized clothes mask to match image dimensions.")
    
    logger.debug(f"Clothes mask shape: {clothes_mask.shape}")
    return clothes_mask

# 옷 영역 보정 함수
def refine_clothes(image: np.array, clothes_mask: np.array) -> np.array:
    refined_image = image.copy()
    logger.debug("Refining clothes area...")

    if clothes_mask.dtype != np.uint8:
        clothes_mask = clothes_mask.astype(np.uint8)

    if clothes_mask.shape[:2] != image.shape[:2]:
        clothes_mask = cv2.resize(clothes_mask, (image.shape[1], image.shape[0]))
        logger.debug("Resized clothes mask to match image dimensions.")

    clothes_region = cv2.bitwise_and(image, image, mask=clothes_mask)
    clothes_region_enhanced = cv2.detailEnhance(clothes_region, sigma_s=10, sigma_r=0.15)
    refined_image[clothes_mask == 255] = clothes_region_enhanced[clothes_mask == 255]

    logger.debug("Clothes area refined successfully.")
    return refined_image

# OpenPose 전처리 함수
def preprocess_openpose(image: np.array) -> np.array:
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.debug("Preprocessing image for OpenPose...")
    results = mp_pose.process(image_rgb)

    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        logger.debug("Pose landmarks drawn on annotated image.")

    pose_image = np.zeros_like(annotated_image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(pose_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        logger.debug("Pose landmarks drawn on pose_image.")

    return pose_image

# Stable Diffusion 업스케일 모델 로드
def load_sd_upscaler():
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    try:
        logger.debug(f"Loading Upscaler model: {model_id}")
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipeline.scheduler.config)
        sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Loaded Upscaler model: {model_id}")
        return sd_pipeline
    except Exception as e:
        logger.error(f"Error loading Upscaler model: {e}", exc_info=True)
        raise RuntimeError(f"Error loading Upscaler model: {e}")

sd_upscaler = load_sd_upscaler()

# 특정 영역을 Stable Diffusion을 사용하여 업스케일하는 함수
def upscale_with_stable_diffusion(image: np.array, region: tuple, sd_upscaler) -> np.array:
    x, y, w, h = region
    logger.debug(f"Upscaling region: x={x}, y={y}, w={w}, h={h}")
    region_image = image[y:y+h, x:x+w]
    
    region_pil = Image.fromarray(cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))
    
    with torch.no_grad():
        upscaled_region = sd_upscaler(
            prompt="", 
            image=region_pil, 
            strength=1.0
        ).images[0]
    
    upscaled_region_np = np.array(upscaled_region)
    upscaled_region_bgr = cv2.cvtColor(upscaled_region_np, cv2.COLOR_RGB2BGR)

    image[y:y+h, x:x+w] = upscaled_region_bgr
    logger.debug("Upscaled region applied to image.")
    
    return image

# 얼굴, 손, 발 영역 업스케일링 함수
def apply_upscale_on_regions(image: np.array, landmarks: dict, min_face_size: int, sd_upscaler) -> np.array:
    logger.debug("Applying upscale on detected regions...")
    if 'face_bbox' in landmarks:
        x, y, w, h = landmarks['face_bbox']
        if w >= min_face_size and h >= min_face_size:
            logger.debug(f"Upscaling face region: x={x}, y={y}, w={w}, h={h}")
            image = upscale_with_stable_diffusion(image, (x, y, w, h), sd_upscaler)
        else:
            logger.debug(f"Face region too small to upscale: w={w}, h={h}")
    return image

# LoRA 파일 목록을 제공하는 엔드포인트
@app.get("/list_lora_files/")
async def list_lora_files():
    try:
        logger.debug(f"Listing LoRA files in: {DEFAULT_LORA_DIR}")
        files = [f for f in os.listdir(DEFAULT_LORA_DIR) if f.endswith('.safetensors')]
        logger.debug(f"Available LoRA files: {files}")
        return {"lora_files": files}
    except Exception as e:
        logger.error(f"Error listing LoRA files: {e}", exc_info=True)
        return JSONResponse(content={"message": f"Error listing LoRA files: {e}"}, status_code=500)

# 이미지 생성 엔드포인트
@app.post("/generate/")
async def generate_image(
    file: UploadFile = File(...), 
    prompt: Optional[str] = Form(None),
    similarity_threshold: float = Form(10.0),
    sketch_mode: bool = Form(False),
    controlnet_strength: float = Form(0.5),
    min_face_size: int = Form(100),
    apply_lora_flag: bool = Form(False),
    lora_file_name: Optional[str] = Form(None)  # LoRA 파일 이름 추가
):
    try:
        logger.debug(f"Received parameters: prompt={prompt}, apply_lora_flag={apply_lora_flag}, min_face_size={min_face_size}, controlnet_strength={controlnet_strength}")

        if not prompt:
            prompt = "a beautiful landscape"  # 기본 프롬프트 설정

        negative_prompt = (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
            "signature, watermark, username, blurry, artist name, bad feet, bad legs, extra legs, "
            "extra limb, extra arm, extra hands, twisted fingers, cut fingers, weird fingers, weird hands, "
            "twisted hands, extra fingers, bad fingers,"
        )

        logger.debug("Opening uploaded image...")
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)
        logger.debug(f"Image size: {image.size}, NumPy shape: {image_np.shape}")

        if sketch_mode:
            logger.debug("Sketch mode enabled.")
            pose_image = preprocess_openpose(image_np)
            controlnet_input = Image.fromarray(pose_image).convert("RGB")
            controlnet = controlnet_models.get('openpose')
            if controlnet is None:
                logger.warning("OpenPose ControlNet 모델이 로드되지 않았습니다.")
                return JSONResponse(content={"message": "OpenPose ControlNet 모델이 로드되지 않았습니다."}, status_code=500)
        else:
            controlnet_input = image
            controlnet = controlnet_models.get('tile')
            if controlnet is None:
                logger.warning("Tile ControlNet 모델이 로드되지 않았습니다.")
                return JSONResponse(content={"message": "Tile ControlNet 모델이 로드되지 않았습니다."}, status_code=500)

        # LoRA 파일 선택 및 적용
        lora_applied = False
        if apply_lora_flag:
            if lora_file_name:
                lora_file_path = os.path.join(DEFAULT_LORA_DIR, lora_file_name)
                if os.path.exists(lora_file_path):
                    logger.debug(f"Applying LoRA model from: {lora_file_path}")
                    lora_applied = load_and_apply_lora(sd_img2img_model, lora_file_path)
                    if not lora_applied:
                        logger.warning("LoRA 적용에 실패했지만, 계속 진행합니다.")
                else:
                    logger.warning(f"LoRA 파일을 찾을 수 없습니다: {lora_file_path}")
                    return JSONResponse(content={"message": f"LoRA 파일을 찾을 수 없습니다: {lora_file_name}"}, status_code=400)
            else:
                logger.warning("LoRA 적용 요청이 되었으나, lora_file_name이 제공되지 않았습니다.")
                return JSONResponse(content={"message": "LoRA 적용을 원하시면 lora_file_name을 제공해야 합니다."}, status_code=400)

        logger.debug("Generating image with Stable Diffusion Img2Img Pipeline...")
        with torch.no_grad():
            generated = sd_img2img_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                controlnet_conditioning_image=controlnet_input,
                controlnet=controlnet,
                strength=controlnet_strength,
                num_inference_steps=20
            ).images[0]
        final_image = np.array(generated)
        logger.debug(f"Generated image shape: {final_image.shape}")

        clothes_mask = detect_clothes(image_np)
        final_image = refine_clothes(final_image, clothes_mask)

        landmarks = detect_face_hands_feet(image_np)
        final_image = apply_upscale_on_regions(final_image, landmarks, min_face_size, sd_upscaler)

        buffered = io.BytesIO()
        Image.fromarray(final_image).save(buffered, format='PNG')
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug("Image converted to base64 string.")

        logger.info("이미지 생성 성공")
        return JSONResponse(content={"message": "이미지 생성 성공", "lora_applied": lora_applied, "image": img_str})

    except RuntimeError as re:
        # 전체 스택 트레이스를 로깅
        logger.error(f"Runtime 오류 발생: {re}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"message": f"Runtime 오류 발생: {re}"}, status_code=500)
    except Exception as e:
        # 전체 스택 트레이스를 로깅
        logger.error(f"이미지 생성 중 일반 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"message": f"이미지 생성 중 오류 발생: {e}"}, status_code=500)

# 텍스트-투-이미지(T2I) 엔드포인트 추가
@app.post("/generate_t2i/")
async def generate_t2i_image(
    prompt: Optional[str] = Form(None),  # prompt가 필수 입력이 아니도록 변경
    apply_lora_flag: bool = Form(False),
    lora_file_name: Optional[str] = Form(None)  # LoRA 파일 이름 추가
):
    try:
        logger.debug(f"Received parameters: prompt={prompt}, apply_lora_flag={apply_lora_flag}")
        
        if not prompt:
            prompt = "a beautiful scene"  # 기본 프롬프트 설정

        negative_prompt = (
            ""
        )

        # LoRA 모델 적용
        lora_applied = False
        logger.debug("Applying LoRA if file name is provided...")
        if apply_lora_flag:
            if lora_file_name:
                lora_file_path = os.path.join(DEFAULT_LORA_DIR, lora_file_name)
                if os.path.exists(lora_file_path):
                    logger.debug(f"Applying LoRA model from: {lora_file_path}")
                    lora_applied = load_and_apply_lora(sd_txt2img_model, lora_file_path)
                    if not lora_applied:
                        logger.warning("LoRA 적용에 실패했지만, 계속 진행합니다.")
                else:
                    logger.warning(f"LoRA 파일을 찾을 수 없습니다: {lora_file_path}")
                    return JSONResponse(content={"message": f"LoRA 파일을 찾을 수 없습니다: {lora_file_name}"}, status_code=400)
            else:
                logger.warning("LoRA 적용 요청이 되었으나, lora_file_name이 제공되지 않았습니다.")
                return JSONResponse(content={"message": "LoRA 적용을 원하시면 lora_file_name을 제공해야 합니다."}, status_code=400)
        else:
            logger.debug("No LoRA model to apply.")

        # 텍스트-투-이미지 생성
        logger.debug("Generating image with Stable Diffusion Txt2Img Pipeline...")
        with torch.no_grad():
            generated_images = sd_txt2img_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20
            )
        
        generated_image = generated_images.images[0]
        final_image = np.array(generated_image)
        logger.debug(f"Generated image shape: {final_image.shape}")

        # 최종 이미지를 저장하여 반환
        buffered = io.BytesIO()
        Image.fromarray(final_image).save(buffered, format='PNG')
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug("Image converted to base64 string.")

        logger.info("텍스트-투-이미지 생성 성공")
        return JSONResponse(content={"message": "텍스트-투-이미지 생성 성공", "lora_applied": lora_applied, "image": img_str})
    
    except RuntimeError as re:
        # 전체 스택 트레이스를 로깅
        logger.error(f"Runtime 오류 발생: {re}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"message": f"Runtime 오류 발생: {re}"}, status_code=500)
    except Exception as e:
        # 전체 스택 트레이스를 로깅
        logger.error(f"텍스트-투-이미지 생성 중 일반 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"message": f"텍스트-투-이미지 생성 중 오류 발생: {e}"}, status_code=500)
