import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import numpy as np
import torch
import torchvision
import os
import shutil
import torchvision.transforms as transforms
from vit_model import ViT
from cvt_model import CvT
from parallel_vit_model import ParallelViT

from PIL import Image

# Show the page title and description.
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁")
st.title("🫁 Lung Cancer Detection")
st.write(
    """
    Lung cancer is a significant contributor to 
    cancer-related mortality. With recent advancements in 
    Computer Vision, Vision Transformers have gained traction 
    and shown remarkable success in medical image analysis. This 
    study explored the potential of Vision Transformer models (ViT, 
    CVT, CCT ViT, Parallel ViT, Efficient ViT) compared to 
    established state-of-the-art architectures (CNN) for lung 
    cancer detection via medical imaging modalities, including CT 
    and Histopathological scans. This work evaluated the impact of 
    data availability and different training approaches on model 
    performance. The training approaches included but were not 
    limited to, Supervised Learning and Transfer Learning. 
    Established evaluation metrics such as accuracy, recall, 
    precision, F1-score, and area under the ROC curve (AUC
    ROC) assessed model performance in terms of detection 
    efficacy, data validity, and computational efficiency. ViT 
    achieved an accuracy of 94% on a balanced dataset and an 
    accuracy of 87% on an imbalanced dataset trained from the 
    ground up. Cost-sensitive evaluation metrics, such as cost 
    matrix and weighted loss, analysed model performance by 
    considering the real-world implications of different types of 
    errors, especially in cases where misdiagnosing a cancer case 
    is far more critical.
    """ 
)

"---"
# --------------------------------------------------------------

# st.subheader("CT Scans of Lung Cancer")
# st.image("images/Lung Cancer Images/CT/CT.png", caption="Sample CT Scan Images Used for Model Training in Lung Cancer Detection")

# st.subheader("Histopathological Images of Lung Cancer")
# st.image("images/Lung Cancer Images/Histopathological/Histopathological.png", caption="Sample  Images Used for Model Training in Lung Cancer Detection")


# --------------------------------------------------------------

<<<<<<< HEAD
st.subheader("** Lung Cancer Image Analysis**")

# Model selection
image_choice = st.selectbox("**Choose the Image Type for Prediction**", options=["CT-Scan Image", "Histopathological Image"])

=======
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
# Define image dimensions and preprocess function based on your model training
IMG_SIZE = (244, 244)  # Match to the input size your model was trained on

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        epsilon = tf.keras.backend.epsilon()  
        return 2 * ((precision * recall) / (precision + recall + epsilon))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
       
def preprocess_cnn_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0) # Add a batch dimension
    return image_array

# Preprocess image for PyTorch ViT models
def preprocess_vit_image(image, target_size):
<<<<<<< HEAD
=======
    from torchvision import transforms

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

def get_vit_input_size(model):
    """Infer expected ViT input size from positional embeddings and patch dimensions."""
    if not hasattr(model, "pos_embedding"):
        return IMG_SIZE

    num_patches = model.pos_embedding.shape[1] - 1
    if num_patches <= 0:
        return IMG_SIZE

    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        return IMG_SIZE

    try:
        patch_module = model.to_patch_embedding[1]
        if hasattr(patch_module, "normalized_shape"):
            patch_dim = patch_module.normalized_shape[0]
        elif hasattr(patch_module, "in_features"):
            patch_dim = patch_module.in_features
        else:
            return IMG_SIZE

        channels = 3
        patch_size = int((patch_dim / channels) ** 0.5)
        image_size = grid_size * patch_size
        return (image_size, image_size)
    except Exception:
        return IMG_SIZE

<<<<<<< HEAD
def print_deduction(status, confidence=None):
    if confidence is not None:
        st.write(f"**Model Confidence:** {confidence:.2f}%")
    if status == 'Benign':
        st.write("**Diagnosis:** The image shows a benign case. No malignancy detected, but regular monitoring is advised.")
    elif status == 'Malignant':
        st.write("**Alert:** The image indicates a malignant lung cancer case. Immediate medical attention is recommended.")
    elif status == 'Normal':
        st.write("**Result:** The image appears normal with no signs of lung cancer.")
    elif status == "Malignant_ACA":
        st.write("**Diagnosis:** The image indicates an Adenocarcinoma (ACA) case. Further evaluation and treatment should be discussed with a healthcare professional.")
    elif status == "Malignant_SCC":
        st.write("**Diagnosis:** The image indicates Squamous Cell Carcinoma (SCC). Prompt medical intervention is necessary, and treatment options should be explored with a specialist.")
=======
def print_deduction(status, confidence=None, output=None):
    target = output if output is not None else st

    severity = "other"
    title = "Prediction"
    message = "Prediction complete."

    if status == 'Benign':
        severity = "benign"
        title = "Benign"
        message = "The image shows a benign case. No malignancy detected, but regular monitoring is advised."
    elif status == 'Malignant':
        severity = "malignant"
        title = "Malignant"
        message = "The image indicates a malignant lung cancer case. Immediate medical attention is recommended."
    elif status == 'Normal':
        severity = "normal"
        title = "Normal"
        message = "The image appears normal with no signs of lung cancer."
    elif status == "Malignant_ACA":
        severity = "malignant"
        title = "Malignant - Adenocarcinoma"
        message = "The image indicates an Adenocarcinoma (ACA) case. Further evaluation and treatment should be discussed with a healthcare professional."
    elif status == "Malignant_SCC":
        severity = "malignant"
        title = "Malignant - Squamous Cell Carcinoma"
        message = "The image indicates Squamous Cell Carcinoma (SCC). Prompt medical intervention is necessary, and treatment options should be explored with a specialist."

    confidence_text = f"{confidence:.2f}%" if confidence is not None else "N/A"
    target.markdown("#### Prediction Result")
    target.markdown(
        f"""
        <div class="result-card {severity}">
            <div class="result-header">
                <div class="result-title">{title}</div>
                <div class="result-confidence">Confidence: {confidence_text}</div>
            </div>
            <div class="result-msg">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)

model_names = ["CNN Base Model", "CNN Hybrid Model", "ViT Base Model", "ViT CVT Model", "ViT Parallel Model"]

# Load Keras models based on user selection
model_paths = {
    "CNN Base Model": "models/cnn_model_2.keras",
    "CNN Hybrid Model": "models/cnn_model_2.keras",
    "ViT Base Model": "models/vit_ground_up_ct_model.pth",
    "ViT CVT Model": "models/vit_cvt_ground_up_ct_model.pth",
    "ViT Parallel Model": "models/vit_parallel_ground_up_ct_model.pth",
    "ViT Histopathological Model": "models/vit_ground_up_histopathological_model.pth"
}

<<<<<<< HEAD
def _build_vit_from_state_dict(state_dict):
    """Recreate the lucidrains-style ViT from a saved state_dict."""
=======
_drive_models_download_attempted = False

def _download_models_folder_from_drive_once():
    """Attempt one-time download of missing artifacts from the shared Drive folder."""
    global _drive_models_download_attempted
    if _drive_models_download_attempted:
        return
    _drive_models_download_attempted = True

    try:
        import gdown
    except ImportError:
        st.error(
            "Missing dependency 'gdown'. Install it with: pip install gdown, "
            "then rerun the app to auto-download models from Google Drive."
        )
        return

    os.makedirs("models", exist_ok=True)
    try:
        with st.spinner("Downloading missing model files from Google Drive. This can take a few minutes on first run..."):
            gdown.download_folder(
                url=DRIVE_MODELS_FOLDER_URL,
                output="models",
                quiet=True,
                remaining_ok=True,
            )
    except Exception as exc:
        st.error(f"Could not download model files from Google Drive: {exc}")

def _ensure_local_artifact(file_path):
    if os.path.exists(file_path):
        return True
    _download_models_folder_from_drive_once()
    if not os.path.exists(file_path):
        st.error(
            f"Missing required file: {file_path}. "
            f"Please verify it exists in the Drive folder: {DRIVE_MODELS_FOLDER_URL}"
        )
        return False
    return True

def _build_vit_from_state_dict(state_dict):
    """Recreate the lucidrains-style ViT from a saved state_dict."""
    from vit_model import ViT

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    if "pos_embedding" not in state_dict or "to_patch_embedding.1.weight" not in state_dict:
        raise ValueError("Unsupported checkpoint format: missing ViT keys.")

    pos_embedding = state_dict["pos_embedding"]
    patch_ln_weight = state_dict["to_patch_embedding.1.weight"]
    mlp_head_weight = state_dict["mlp_head.weight"]

    num_patches = pos_embedding.shape[1] - 1
    dim = pos_embedding.shape[2]
    patch_dim = patch_ln_weight.shape[0]
    channels = 3
    patch_size = int((patch_dim / channels) ** 0.5)
    image_size = int((num_patches ** 0.5) * patch_size)

    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.append(int(parts[2]))
    if not layer_indices:
        raise ValueError("Unsupported checkpoint format: cannot infer transformer depth.")
    depth = max(layer_indices) + 1

    qkv_weight = state_dict["transformer.layers.0.0.to_qkv.weight"]
    inner_dim = qkv_weight.shape[0] // 3
    dim_head = 64
    heads = max(1, inner_dim // dim_head)

    mlp_dim = state_dict["transformer.layers.0.1.net.1.weight"].shape[0]
    num_classes = mlp_head_weight.shape[0]

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dim_head=dim_head
    )
    model.load_state_dict(state_dict)
    return model

def _build_cvt_from_state_dict(state_dict):
    """Recreate the project CvT variant from a saved state_dict."""
<<<<<<< HEAD
=======
    from cvt_model import CvT

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    num_classes = state_dict["to_logits.2.weight"].shape[0]

    model = CvT(
        num_classes=num_classes,
        s1_emb_dim=128,
        s1_emb_kernel=7,
        s1_emb_stride=4,
        s1_proj_kernel=3,
        s1_kv_proj_stride=2,
        s1_heads=2,
        s1_depth=2,
        s1_mlp_mult=4,
        s2_emb_dim=256,
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=4,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=512,
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=10,
        s3_depth=2,
        s3_mlp_mult=4,
        dropout=0.1,
        channels=3
    )
    model.load_state_dict(state_dict)
    return model

def _build_parallel_vit_from_state_dict(state_dict):
    """Recreate the project ParallelViT variant from a saved state_dict."""
<<<<<<< HEAD
=======
    from parallel_vit_model import ParallelViT

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    pos_embedding = state_dict["pos_embedding"]
    num_patches = pos_embedding.shape[1] - 1
    dim = pos_embedding.shape[2]

    patch_linear_weight = state_dict["to_patch_embedding.1.weight"]
    patch_dim = patch_linear_weight.shape[1]
    channels = 3
    patch_size = int((patch_dim / channels) ** 0.5)
    image_size = int((num_patches ** 0.5) * patch_size)

    layer_indices = []
    branch_indices = []
    for key in state_dict.keys():
        if not key.startswith("transformer.layers."):
            continue

        parts = key.split(".")
        # Expected pattern examples:
        # transformer.layers.<layer_idx>.<block_idx>.fns.<branch_idx>.to_qkv.weight
        # transformer.layers.<layer_idx>.<block_idx>.fns.<branch_idx>.net.1.weight
        if len(parts) > 2 and parts[2].isdigit():
            layer_indices.append(int(parts[2]))

        if len(parts) > 5 and parts[4] == "fns" and parts[5].isdigit():
            branch_indices.append(int(parts[5]))

    if not layer_indices:
        raise ValueError("Unsupported ParallelViT checkpoint format.")

    depth = max(layer_indices) + 1
    num_parallel_branches = max(branch_indices) + 1 if branch_indices else 2

    qkv_weight = state_dict["transformer.layers.0.0.fns.0.to_qkv.weight"]
    inner_dim = qkv_weight.shape[0] // 3
    dim_head = 64
    heads = max(1, inner_dim // dim_head)

    mlp_dim = state_dict["transformer.layers.0.1.fns.0.net.1.weight"].shape[0]
    num_classes = state_dict["mlp_head.1.weight"].shape[0]

    model = ParallelViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        num_parallel_branches=num_parallel_branches,
        channels=channels,
        dim_head=dim_head,
        dropout=0.1,
        emb_dropout=0.1
    )
    model.load_state_dict(state_dict)
    return model

def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return loaded_obj["state_dict"]
        if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            return loaded_obj["model_state_dict"]
        return loaded_obj
    return None

def _build_model_from_state_dict(state_dict):
    state_keys = set(state_dict.keys())

    if "transformer.layers.0.0.fns.0.to_qkv.weight" in state_keys:
        return _build_parallel_vit_from_state_dict(state_dict)

    if "pos_embedding" in state_keys and "to_patch_embedding.1.weight" in state_keys:
        return _build_vit_from_state_dict(state_dict)

    if "to_logits.2.weight" in state_keys and any(k.startswith("layers.") for k in state_keys):
        return _build_cvt_from_state_dict(state_dict)

    raise ValueError("Unsupported checkpoint format for configured models.")

def _materialize_lfs_pointer_if_possible(file_path):
    """Replace a Git LFS pointer file with its local object if already downloaded."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [f.readline().strip() for _ in range(3)]
    except Exception:
        return

    if not lines or not lines[0].startswith("version https://git-lfs.github.com/spec/v1"):
        return

    oid_line = next((line for line in lines if line.startswith("oid sha256:")), None)
    if not oid_line:
        return

    oid = oid_line.split("oid sha256:", 1)[1].strip()
    if len(oid) < 4:
        return

    lfs_object_path = os.path.join(".git", "lfs", "objects", oid[:2], oid[2:4], oid)
    if not os.path.exists(lfs_object_path):
        return

    if os.path.getsize(file_path) < os.path.getsize(lfs_object_path):
        shutil.copyfile(lfs_object_path, file_path)

<<<<<<< HEAD
@st.cache_data
=======
@st.cache_resource(show_spinner=False)
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
def loadModel(model_name, model_signature=None):
    model_path = model_paths.get(model_name)

    if not model_path:
        st.error(f"Unknown model selection: '{model_name}'.")
        return None, None
        
<<<<<<< HEAD
    if model_path and os.path.exists(model_path):
        if model_path.endswith(".keras"):
            # Load TensorFlow model
        
            model = tf.keras.models.load_model('models//cnn_model_2.keras')
=======
    if model_path and _ensure_local_artifact(model_path):
        if model_path.endswith(".keras"):
            # Load TensorFlow model

            model = tf.keras.models.load_model(model_path)
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
            return model, 'tf'
        
        elif model_path.endswith(".pth"):
            # Load PyTorch model
            # Ensure CUDA-saved checkpoints can be loaded on CPU-only machines.
            try:
<<<<<<< HEAD
=======
                import torch

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
                _materialize_lfs_pointer_if_possible(model_path)
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    loaded_obj = torch.load(model_path, map_location=target_device)
                except Exception:
                    # PyTorch>=2.6 defaults weights_only=True; retry for trusted local checkpoints.
                    loaded_obj = torch.load(model_path, map_location=target_device, weights_only=False)

                if isinstance(loaded_obj, torch.nn.Module):
                    model = loaded_obj
                else:
                    state_dict = _extract_state_dict(loaded_obj)
                    if isinstance(state_dict, dict):
                        model = _build_model_from_state_dict(state_dict)
                    else:
                        raise TypeError(f"Unsupported checkpoint type: {type(loaded_obj)}")

                model.to(target_device)
                model.eval()  # Set model to evaluation mode
                return model, 'torch'
            except Exception as exc:
                exc_msg = str(exc)
                if "invalid load key, 'v'" in exc_msg:
                    exc_msg = (
                        "Checkpoint file is not a valid PyTorch binary (likely a Git LFS pointer). "
                        "Please download the real .pth weights file."
                    )
                st.error(f"Failed to load PyTorch model '{model_name}': {exc_msg}")
                return None, None
    else:
        st.error(f"Model file not found: {model_path}")
            
    return None, None
    
# Run model on uploaded image
<<<<<<< HEAD
def run_model(model_name, image):
=======
def run_model(model_name, image, output=None):
    target = output if output is not None else st

    loader_slot = target.empty()
    loader_slot.markdown(
        """
        <div class="inference-loader">
            <div class="inference-loader-text">Running AI inference pipeline...</div>
            <div class="inference-loader-track">
                <div class="inference-loader-bar"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    model_path = model_paths.get(model_name)
    model_signature = None
    if model_path and os.path.exists(model_path):
        model_signature = (os.path.getmtime(model_path), os.path.getsize(model_path))

<<<<<<< HEAD
    model, framework = loadModel(model_name, model_signature)
    if model:
        if framework == 'tf':
            # Preprocess and predict using TensorFlow model
            processed_image = preprocess_cnn_image(image)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100  # Get confidence as a percentage
            
        elif framework == 'torch':
            # Preprocess and predict using PyTorch model
            vit_input_size = get_vit_input_size(model)
            processed_image = preprocess_vit_image(image, vit_input_size)
            model_device = next(model.parameters()).device
            processed_image = processed_image.to(model_device)
            with torch.no_grad():
                predictions = model(processed_image)
            predicted_class = predictions.argmax(dim=1).item()
            confidence = torch.softmax(predictions, dim=1)[0, predicted_class].item() * 100  # Confidence for PyTorch

        # Define class labels (adjust these to match your model's output)
        class_labels = ["Normal", "Benign", "Malignant", "Malignant_ACA", "Malignant_SCC"]
        status = class_labels[predicted_class]

        # Display the result with confidence
        print_deduction(status, confidence)
    else:
        st.error(f"Model '{model_name}' could not be loaded.")

if image_choice == "CT-Scan Image":
    model_choice = st.selectbox("**Choose a Model for Prediction**", options=sorted(model_names))
elif image_choice == "Histopathological Image":
    model_choice = st.selectbox("**Choose a Model for Prediction**", options=["ViT Histopathological Model"])

# Title
st.subheader("**Upload an Image**")

# Image uploader
uploaded_file = st.file_uploader("**Choose an image...**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a "Predict" button
    if st.button("Predict Image"):
        # Run model on uploaded image
        run_model(model_choice, image)
=======
    try:
        model, framework = loadModel(model_name, model_signature)
        if model:
            if framework == 'tf':
                # Preprocess and predict using TensorFlow model
                processed_image = preprocess_cnn_image(image)
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions) * 100  # Get confidence as a percentage

            elif framework == 'torch':
                # Preprocess and predict using PyTorch model
                import torch

                vit_input_size = get_vit_input_size(model)
                processed_image = preprocess_vit_image(image, vit_input_size)
                model_device = next(model.parameters()).device
                processed_image = processed_image.to(model_device)
                with torch.no_grad():
                    predictions = model(processed_image)
                predicted_class = predictions.argmax(dim=1).item()
                confidence = torch.softmax(predictions, dim=1)[0, predicted_class].item() * 100  # Confidence for PyTorch

            # Define class labels (adjust these to match your model's output)
            class_labels = ["Normal", "Benign", "Malignant", "Malignant_ACA", "Malignant_SCC"]
            status = class_labels[predicted_class]

            # Display the result with confidence
            print_deduction(status, confidence, output=output)
        else:
            target.error(f"Model '{model_name}' could not be loaded.")
    finally:
        loader_slot.empty()

st.markdown("<div class='section-banner'><span class='section-dot'></span>Clinical Workspace</div>", unsafe_allow_html=True)
image_col, survey_col = st.columns([1.2, 1.0], gap="large")

uploaded_file = None
selected_image_model = None
predict_image_clicked = False

with image_col:
    st.markdown("#### Image Model Controls")
    image_choice = st.selectbox(
        "**Choose Image Type**",
        options=["CT-Scan Image", "Histopathological Image"],
        key="img_type"
    )

    if image_choice == "CT-Scan Image":
        selected_image_model = st.selectbox(
            "**Choose Model**",
            options=sorted(model_names),
            key="img_model_ct"
        )
    else:
        selected_image_model = st.selectbox(
            "**Choose Model**",
            options=["ViT Histopathological Model"],
            key="img_model_hist"
        )

    uploaded_file = st.file_uploader(
        "**Choose an image...**",
        type=["jpg", "jpeg", "png"],
        key="img_upload"
    )

    selected_model_path = model_paths.get(selected_image_model, "")
    model_ready = os.path.exists(selected_model_path)
    model_framework = "TensorFlow" if selected_model_path.endswith(".keras") else "PyTorch"
    runtime_device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    st.markdown(
        f"""
        <div class="model-health-row">
            <div class="health-badge {'health-ready' if model_ready else 'health-warn'}">
                <span class="health-dot"></span>
                {'Local Model Ready' if model_ready else 'Model will fetch from Drive'}
            </div>
            <div class="health-badge {'health-gpu' if runtime_device == 'GPU' else 'health-cpu'}">
                <span class="health-dot"></span>
                Runtime: {runtime_device}
            </div>
            <div class="health-badge health-ready">
                <span class="health-dot"></span>
                Framework: {model_framework}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_image_clicked = st.button("Predict Image", key="predict_image_button")

    prediction_output = st.container()
    if predict_image_clicked:
        if uploaded_file is None:
            with prediction_output:
                st.markdown("#### Prediction Result")
                st.warning("Please upload an image first.")
        else:
            image_for_prediction = Image.open(uploaded_file)
            run_model(selected_image_model, image_for_prediction, output=prediction_output)
    else:
        with prediction_output:
            if uploaded_file is None:
                st.markdown("#### Prediction Result")
                st.caption("Upload an image and click Predict Image to view results here.")

    st.markdown("#### Preview and Diagnosis")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=420)
    else:
        st.info("Upload an image to preview here.")
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)

"---"
# --------------------------------------------------------------

st.subheader("🔍 Exploring Lung Cancer")
st.write(
    """
    This section visualises data from the [Exploring Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data).
    The effectiveness of cancer prediction system can inform individuals of their cancer risk with low cost and it will help people to make a more informed decision based on their cancer risk status. 
    Just click on the widgets below to explore!
    """
)

cancer_directory = "data/survey_lung_cancer.csv"
@st.cache_data
def load_data():
    lung_df = pd.read_csv(cancer_directory)
    lung_df.columns = lung_df.columns.str.replace('_', ' ').str.strip().str.title()
    return lung_df

lung_df = load_data()

<<<<<<< HEAD

print(lung_df.columns)

=======
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
# Mapping dictionary for binary columns (1: No, 2: Yes)
binary_mapping = {1: "No", 2: "Yes", "YES": "Yes", "NO": "No", "M": "Male", "F": "Female"}

columns = ["Gender", "Age"]

# Apply mapping to each binary column
for col in lung_df.columns:
    if col != "Age":
        lung_df[col] = lung_df[col].map(binary_mapping)

# Gender selection with mapped values
genders = st.multiselect(
    "**Select Gender**",
    options=lung_df["Gender"].unique().tolist(),
    default=["Male", "Female"]
)

main_features = ["Smoking", "Peer Pressure", "Chronic Disease", "Alcohol Consuming"]
main_symptoms = ["Yellow Fingers", "Anxiety", "Fatigue", "Allergy", "Wheezing", "Coughing", "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"]

# Features multiselect with relevant features
features = st.multiselect(
    "**Select Features**",
    options=main_features,
    default=main_features
)

# Symptoms multiselect based on symptom columns
symptoms = st.multiselect(
    "**Select Symptoms**",
    options=main_symptoms,
    default=main_symptoms
)

# Age slider based on the dataset's age range (1-120)
ages = st.slider(
    "**Select Age Range**", 
    min_value=1, 
    max_value=120, 
    value=(20, 50)
)

# Filter the dataframe based on widget inputs
lung_df_filtered = lung_df[
    (lung_df["Gender"].isin(genders)) &
    (lung_df["Age"].between(ages[0], ages[1])) 
]

lung_df_filtered = lung_df_filtered.sort_values(by="Age", ascending=True)

# Select only the necessary columns based on user input
columns_to_display = ["Age", "Gender", "Lung Cancer"] + features + symptoms
lung_df_filtered = lung_df_filtered[columns_to_display]

st.dataframe(
    lung_df_filtered,
<<<<<<< HEAD
    use_container_width=True,
=======
    width='stretch',
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    column_config={"Age": st.column_config.TextColumn("Age")},
)

# --------------------------------------------------------------
# LUNG CANCER STATISTICS

"---"

<<<<<<< HEAD
# Pie chart for Lung Cancer status
lung_cancer_counts = lung_df_filtered['Lung Cancer'].value_counts().reset_index()
lung_cancer_counts.columns = ['Status', 'Count']

lung_cancer_chart = (
    alt.Chart(lung_cancer_counts)
    .mark_arc(innerRadius=50, stroke='white')
    .encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", legend=alt.Legend(title="Lung Cancer Status")),
        tooltip=['Status', 'Count']
    )
    .properties(title="Lung Cancer Status Distribution")
)

st.altair_chart(lung_cancer_chart, use_container_width=True)

"---"

# Bar chart for Gender distribution
gender_counts = lung_df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

gender_chart = (
    alt.Chart(gender_counts)
    .mark_bar()
    .encode(
        x=alt.X('Gender:N', title='Gender'),
        y=alt.Y('Count:Q', title='Count'),
        color='Gender:N',
        tooltip=['Gender', 'Count']
    )
    .properties(title="Gender Distribution")
)

st.altair_chart(gender_chart, use_container_width=True)
=======
st.markdown(
    """
    <div class="viz-section">
        <div class="viz-title">Interactive Cancer Analytics</div>
        <div class="viz-sub">Hover over chart elements to spotlight key trends and compare distributions instantly.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pie chart for Lung Cancer status
lung_cancer_counts = lung_df_filtered['Lung Cancer'].value_counts().reset_index()
lung_cancer_counts.columns = ['Status', 'Count']
lung_cancer_counts['Percent'] = (lung_cancer_counts['Count'] / lung_cancer_counts['Count'].sum()) * 100

pie_highlight = alt.selection_point(fields=['Status'], on='mouseover', empty=True)

lung_cancer_chart = (
    alt.Chart(lung_cancer_counts)
    .mark_arc(innerRadius=68, cornerRadius=7, padAngle=0.02, stroke='#081320', strokeWidth=2)
    .encode(
        theta=alt.Theta(field='Count', type='quantitative'),
        color=alt.Color(
            field='Status',
            type='nominal',
            legend=alt.Legend(title='Lung Cancer Status'),
            scale=alt.Scale(range=['#2dd4bf', '#0ea5e9', '#f87171'])
        ),
        opacity=alt.condition(pie_highlight, alt.value(1), alt.value(0.65)),
        tooltip=[
            alt.Tooltip('Status:N', title='Status'),
            alt.Tooltip('Count:Q', title='Cases'),
            alt.Tooltip('Percent:Q', title='Share', format='.1f')
        ]
    )
    .add_params(pie_highlight)
    .properties(title='Lung Cancer Status Distribution', height=320)
    .configure(background='transparent')
    .configure_view(strokeOpacity=0)
    .configure_title(color='#e6f4ff', fontSize=18)
    .configure_axis(labelColor='#c3d8ea', titleColor='#e6f4ff', gridColor='#29405a')
    .configure_legend(labelColor='#c3d8ea', titleColor='#e6f4ff')
)

gender_counts = lung_df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

bar_highlight = alt.selection_point(fields=['Gender'], on='mouseover', empty=True)

bar_base = alt.Chart(gender_counts).encode(
    x=alt.X('Gender:N', title='Gender', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Count:Q', title='Count'),
    tooltip=[alt.Tooltip('Gender:N'), alt.Tooltip('Count:Q')]
)

bar_chart = bar_base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
    color=alt.condition(
        bar_highlight,
        alt.Color('Gender:N', scale=alt.Scale(range=['#2dd4bf', '#0ea5e9']), legend=None),
        alt.value('#3a5875')
    )
).add_params(bar_highlight)

bar_labels = bar_base.mark_text(
    dy=-8,
    color='#dff6ff',
    fontSize=12,
    fontWeight='bold'
).encode(text='Count:Q')

gender_chart = (
    (bar_chart + bar_labels)
    .properties(title='Gender Distribution', height=320)
    .configure(background='transparent')
    .configure_view(strokeOpacity=0)
    .configure_title(color='#e6f4ff', fontSize=18)
    .configure_axis(labelColor='#c3d8ea', titleColor='#e6f4ff', gridColor='#29405a')
    .configure_legend(labelColor='#c3d8ea', titleColor='#e6f4ff')
)

chart_col_1, chart_col_2 = st.columns(2, gap='large')

with chart_col_1:
    
    st.altair_chart(lung_cancer_chart, width='stretch')
    st.markdown("<div class='chart-caption'>Status ratio is shown as a donut chart with hover spotlight.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col_2:
    
    st.altair_chart(gender_chart, width='stretch')
    st.markdown("<div class='chart-caption'>Hover bars to emphasize each cohort and compare case volume.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)

"---"
# --------------------------------------------------------------

<<<<<<< HEAD
# Streamlit UI
st.subheader("📋 Lung Cancer Prediction Survey")
st.write("**Enter the patient's information below to predict the likelihood of lung cancer:**")

@st.cache_data
def load_models():
    # Load the models
    lr_model = joblib.load('models/lr_model.pkl')  # Corrected model name
=======
@st.cache_resource(show_spinner=False)
def load_models():
    # Load the models
    aux_paths = [
        'models/lr_model.pkl',
        'models/knn_model.pkl',
        'models/label_encoder.pkl',
        'models/scaler.pkl',
    ]
    for aux_path in aux_paths:
        if not _ensure_local_artifact(aux_path):
            st.error(f"Required model artifact not found: {aux_path}")
            st.stop()

    lr_model = joblib.load('models/lr_model.pkl')
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
    knn_model = joblib.load('models/knn_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return lr_model, knn_model, label_encoder, scaler

<<<<<<< HEAD
lr_model, knn_model, label_encoder, scaler = load_models()

# Age input
age = st.slider("**Select Age**", min_value=1, max_value=120, value=30)

# Gender selection
gender = st.selectbox("**Select Gender**", options=["Male", "Female"])

# User input for binary features
feature_inputs = {}
for feature in lung_df.columns:
    if feature not in columns and feature != "Lung Cancer":
        feature_inputs[feature] = st.selectbox(f"**{feature}?**", options=["No", "Yes"])

# Model selection
model_choice = st.selectbox("**Choose a Model for Prediction**", options=["Logistic Regression", "K-Nearest Neighbors"])
selected_model = lr_model if model_choice == "Logistic Regression" else knn_model

if st.button("Predict"):
    # Prepare input data for prediction
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        **feature_inputs,
        "Lung Cancer": "No"
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col not in columns:
            input_df[col] = label_encoder.transform(input_df[col])

    # Transform features
    del input_df["Lung Cancer"]  
    input_df = scaler.transform(input_df)

    # Prediction
    prediction = selected_model.predict(input_df)
    result = "Likely to have lung cancer." if prediction[0] == 1 else "Unlikely to have lung cancer."
    # Display prediction result
    st.write("\n\n**Prediction Result:**", result)
=======

with survey_col:
    st.markdown("#### 📋 Survey Prediction")
    st.caption("Compact risk-form controls for quick patient screening.")

    age = st.slider("**Select Age**", min_value=1, max_value=120, value=30, key="survey_age")
    gender = st.selectbox("**Select Gender**", options=["Male", "Female"], key="survey_gender")

    feature_inputs = {}
    survey_features = [
        feature for feature in lung_df.columns
        if feature not in columns and feature != "Lung Cancer"
    ]

    feat_col_1, feat_col_2 = st.columns(2, gap="small")
    for idx, feature in enumerate(survey_features):
        target_col = feat_col_1 if idx % 2 == 0 else feat_col_2
        with target_col:
            feature_inputs[feature] = st.selectbox(
                f"**{feature}?**",
                options=["No", "Yes"],
                key=f"survey_feature_{idx}"
            )

    survey_model_choice = st.selectbox(
        "**Choose Model**",
        options=["Logistic Regression", "K-Nearest Neighbors"],
        key="survey_model_choice"
    )

    if st.button("Predict Survey", key="predict_survey_button"):
        lr_model, knn_model, label_encoder, scaler = load_models()
        selected_model = lr_model if survey_model_choice == "Logistic Regression" else knn_model

        input_data = {
            "Gender": 1 if gender == "Male" else 0,
            "Age": age,
            **feature_inputs,
            "Lung Cancer": "No"
        }

        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if col not in columns:
                input_df[col] = label_encoder.transform(input_df[col])

        del input_df["Lung Cancer"]
        input_df = scaler.transform(input_df)

        prediction = selected_model.predict(input_df)
        result = "Likely to have lung cancer." if prediction[0] == 1 else "Unlikely to have lung cancer."
        st.success(result)
>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)

# --------------------------------------------------------------


<<<<<<< HEAD
# logo and images

UKZN_LOGO = "images/UKZN.png"
st.logo(
    UKZN_LOGO,
    icon_image=UKZN_LOGO,
    size="large"
)

# make logo vanish when scrolling down
st.markdown(
    """
    <style>
    img[data-testid="stLogo"] {
        height: 4rem;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------


"---"
st.subheader("🌍 Lung Cancer Research")


"---"
=======
st.subheader("🌍 Lung Cancer Research")



>>>>>>> 4342f23 (Initial commit: Lung Cancer Detection with ViT)
st.subheader("🔗 References")

st.write(
    "- **[Lung Cancer DataSet](https://www.kaggle.com/datasets/yusufdede/lung-cancer-dataset), Yusuf Dede (2018)**"
)

st.write(
    "- **[Lung and Colon Cancer Histopathological Image Dataset (LC25000)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data), Borkowski AA (2019)**"
)

st.write(
    "- **[The IQ-OTH/NCCD lung cancer dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/data), Alyasriy (2023)**"
)