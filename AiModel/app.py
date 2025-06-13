import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import io
import datetime
# ==============================================================================
# Page Configuration (must be the first Streamlit command)
# ==============================================================================
st.set_page_config(
    page_title="Fundus Classifier with Grad-CAM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Custom CSS for Styling
# ==============================================================================
st.markdown("""
    <style>
        /* General layout and font styling */
        .main > div {max-width: 1200px; padding-top: 1rem;}
        h1, h2, h3 {margin-top: 0; margin-bottom: 0.5rem; color: #2C7A7B;}
        div.block-container {padding-top: 1rem; padding-bottom: 1rem;}
        section.main > div:has(~ footer) {padding-bottom: 1rem;}
        
        /* Make images smaller */
        .stImage > img {
            max-height: 200px; /* Reduced from 250px */
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        
        /* Style the primary button */
        div.stButton > button {background-color: #2C7A7B; color: white;}
        
        /* Style for prediction boxes */
        .prediction-container {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 10px;
            border: 1px solid #ccc;
        }
        .prediction-label {
            font-weight: bold;
            font-size: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: #333; /* Darker text for readability */
        }
        .prediction-confidence {
            font-size: 0.8rem;
            color: #555;
            margin-top: 4px;
        }
        
        /* Image container with info icon */
        .image-container {
            position: relative;
        }
        .info-icon {
            position: absolute;
            top: 5px;
            right: 5px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 50%;
            padding: 3px;
            cursor: pointer;
            z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Configuration and Constants
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 299
# IMPORTANT: Replace this with the actual path to your model file
MODEL_PATH = r'D:/Kaif/Hackathon25/CDAC/AiModel/keggle/best_fundus_efficientnetb3.pth' 
CLASS_NAMES = sorted([
    '0.0.Normal', '0.1.Tessellated fundus', '0.2.Large optic cup', '0.3.DR1', '1.0.DR2', '1.1.DR3',
    '10.0.Possible glaucoma', '10.1.Optic atrophy', '11.Severe hypertensive retinopathy',
    '12.Disc swelling and elevation', '13.Dragged Disc', '14.Congenital disc abnormality',
    '15.0.Retinitis pigmentosa', '15.1.Bietti crystalline dystrophy',
    '16.Peripheral retinal degeneration and break', '17.Myelinated nerve fiber', '18.Vitreous particles',
    '19.Fundus neoplasm', '2.0.BRVO', '2.1.CRVO', '20.Massive hard exudates',
    '21.Yellow-white spots-flecks', '22.Cotton-wool spots', '23.Vessel tortuosity',
    '24.Chorioretinal atrophy-coloboma', '25.Preretinal hemorrhage', '26.Fibrosis', '27.Laser Spots',
    '28.Silicon oil in eye', '29.0.Blur fundus without PDR', '29.1.Blur fundus with suspected PDR',
    '3.RAO', '4.Rhegmatogenous RD', '5.0.CSCR', '5.1.VKH disease', '6.Maculopathy', '7.ERM', '8.MH',
    '9.Pathological myopia', '1000images'
])

# Disease Information Database
DISEASE_INFO = {
    'Normal': {
        'description': 'The fundus appears healthy with no signs of pathological changes.',
        'symptoms': ['No symptoms', 'Regular vision', 'No visual disturbances']
    },
    'Tessellated fundus': {
        'description': 'A benign condition where the choroidal blood vessels are visible through a thin retinal pigment epithelium.',
        'symptoms': ['Usually asymptomatic', 'May have mild visual changes', 'No pain or discomfort']
    },
    'Large optic cup': {
        'description': 'An enlarged optic cup which may indicate glaucoma or other optic nerve conditions.',
        'symptoms': ['Gradual vision loss', 'Peripheral vision loss', 'Difficulty seeing in low light']
    },
    'DR1': {
        'description': 'Mild non-proliferative diabetic retinopathy - early stage of diabetic eye disease.',
        'symptoms': ['Often no symptoms', 'Occasional blurred vision', 'Difficulty focusing']
    },
    'DR2': {
        'description': 'Moderate non-proliferative diabetic retinopathy with more advanced retinal changes.',
        'symptoms': ['Blurred vision', 'Dark spots in vision', 'Difficulty reading']
    },
    'DR3': {
        'description': 'Severe non-proliferative diabetic retinopathy requiring immediate attention.',
        'symptoms': ['Significant vision loss', 'Floaters', 'Dark areas in vision', 'Difficulty with night vision']
    },
    'Possible glaucoma': {
        'description': 'Signs suggesting glaucoma, a condition that damages the optic nerve.',
        'symptoms': ['Gradual peripheral vision loss', 'Eye pain', 'Halos around lights', 'Headaches']
    },
    'Optic atrophy': {
        'description': 'Damage or deterioration of the optic nerve.',
        'symptoms': ['Vision loss', 'Reduced color vision', 'Decreased visual field', 'Poor night vision']
    },
    'Severe hypertensive retinopathy': {
        'description': 'Severe damage to retinal blood vessels due to high blood pressure.',
        'symptoms': ['Sudden vision loss', 'Headaches', 'Double vision', 'Dim vision']
    },
    'Disc swelling and elevation': {
        'description': 'Swelling of the optic disc, which may indicate increased intracranial pressure.',
        'symptoms': ['Headaches', 'Vision changes', 'Nausea', 'Double vision']
    },
    'Maculopathy': {
        'description': 'Disease affecting the macula, the central part of the retina.',
        'symptoms': ['Central vision loss', 'Distorted vision', 'Difficulty reading', 'Problems recognizing faces']
    },
    'Pathological myopia': {
        'description': 'Severe nearsightedness that can lead to retinal complications.',
        'symptoms': ['Severe nearsightedness', 'Floaters', 'Flashing lights', 'Distorted vision']
    }
}

def get_disease_info(label):
    """Get disease information and symptoms for a given label"""
    # Clean the label to match our database keys
    clean_label = label.replace('_', ' ').strip()
    
    # Try to find exact match first
    if clean_label in DISEASE_INFO:
        return DISEASE_INFO[clean_label]
    
    # Try to find partial match
    for key in DISEASE_INFO.keys():
        if key.lower() in clean_label.lower() or clean_label.lower() in key.lower():
            return DISEASE_INFO[key]
    
    # Default information if no match found
    return {
        'description': 'This condition requires further evaluation by an ophthalmologist for proper diagnosis and treatment.',
        'symptoms': ['Consult with an eye care professional', 'Regular eye examinations recommended', 'Monitor any vision changes']
    }

# Image info tooltips
ORIGINAL_IMG_INFO = """
**Original Fundus Image**

This is the uploaded retinal fundus photograph that shows the interior surface of the eye, 
including the retina, optic disc, macula, and blood vessels. The fundus image is used by 
ophthalmologists to diagnose various eye conditions.
"""

GRADCAM_IMG_INFO = """
**Grad-CAM Visualization**

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of the image 
that were most important for the model's prediction. Red areas indicate regions that 
strongly influenced the classification decision, helping to explain what the AI model 
is "looking at" when making its diagnosis.
"""

# ==============================================================================
# Image Transformations
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ==============================================================================
# Model Loading
# ==============================================================================
@st.cache_resource
def load_model():
    """Loads the pre-trained model. Caches the resource for performance."""
    try:
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_gradient_color(confidence):
    """Creates a gradient color from light green to dark teal based on confidence."""
    start_r, start_g, start_b = 227, 252, 247
    end_r, end_g, end_b = 44, 122, 123
    r = int(start_r + (end_r - start_r) * confidence)
    g = int(start_g + (end_g - start_g) * confidence)
    b = int(start_b + (end_b - start_b) * confidence)
    return f'rgb({r}, {g}, {b})'

def process_and_predict(image: Image.Image):
    """
    Processes an image, runs prediction, and generates Grad-CAM overlay.
    This function now uses the logic from your provided script.
    """
    model = load_model()
    cam_extractor = GradCAM(model, target_layer="conv_head")
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Forward pass to get raw scores (logits). Gradients are not disabled
    # as they are required by the cam_extractor.
    scores = model(input_tensor)
    
    # **MODIFIED: Use sigmoid to get probabilities, as in your example script.**
    # This is suitable for multi-label scenarios.
    probs = torch.sigmoid(scores).cpu().detach().numpy()[0]
    
    # Get the top 5 predictions for display
    top5_idx = probs.argsort()[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_labels = [CLASS_NAMES[i].split('.')[-1].replace('_', ' ') for i in top5_idx]

    # **MODIFIED: Get the single top predicted class index for Grad-CAM generation.**
    predicted_idx = int(top5_idx[0])

    # Generate Grad-CAM for the top predicted class
    cam_map = cam_extractor(class_idx=predicted_idx, scores=scores)[0].cpu().squeeze()

    # Re-normalize the input image for display
    display_img = input_tensor.squeeze().cpu()
    display_img = torch.clamp((display_img * 0.5) + 0.5, 0, 1)
    img_pil = to_pil_image(display_img)

    # Overlay the heatmap on the original image
    cam_result = overlay_mask(img_pil, to_pil_image(cam_map, mode='F'), alpha=0.5)

    return img_pil, cam_result, list(zip(top5_labels, top5_probs))

def create_report_text(predictions, filename):
    """Generates a formatted text string for the report."""
    report = f"Fundus Image Analysis Report\n"
    report += f"============================\n"
    report += f"File Name: {filename}\n"
    report += f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "Top 5 Predictions (Label: Confidence):\n"
    report += "--------------------------------------\n"
    for label, conf in predictions:
        # Note: Sigmoid outputs are independent probabilities, not summing to 1.
        # Formatting as percentage is still intuitive for users.
        report += f"- {label}: {conf:.2%}\n"
    
    report += "\nDisclaimer: This is an automated analysis and not a substitute for professional medical advice."
    return report

def display_image_with_info_icon(image, title, info_text):
    """Display an image with an info icon in the top right corner"""
    # Create a unique key for this tooltip
    tooltip_key = f"tooltip_{title.replace(' ', '_')}_{datetime.datetime.now().timestamp()}"
    
    # Create a container for the image with relative positioning
    st.markdown(f"""
        <div class="image-container">
            <div class="info-icon" onclick="
                if (document.getElementById('{tooltip_key}').style.display === 'none') {{
                    document.getElementById('{tooltip_key}').style.display = 'block';
                }} else {{
                    document.getElementById('{tooltip_key}').style.display = 'none';
                }}
            ">ℹ️</div>
            <div id="{tooltip_key}" style="display: none; position: absolute; top: 30px; right: 5px; 
                background-color: white; border: 1px solid #ddd; border-radius: 5px; 
                padding: 10px; width: 250px; z-index: 1000; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                {info_text}
                <div style="text-align: right; margin-top: 5px;">
                    <small onclick="document.getElementById('{tooltip_key}').style.display = 'none'" 
                    style="cursor: pointer; color: #2C7A7B;">Close</small>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Display the image with caption
    st.image(image, caption=title, use_column_width=True)

# ==============================================================================
# Main User Interface
# ==============================================================================
# Header
st.markdown("<h1 style='text-align: center;'>Fundus Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-top: -0.5rem;'>Visual Explanation using Grad-CAM</p>", unsafe_allow_html=True)

# Create tabs for main content and a placeholder for a future GenAI feature
tab1, tab2 = st.tabs(["**🔬 Analysis**", "**🤖 GenAI Assistant**"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fundus image (JPG, PNG)...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 2. Run Analysis")
        analyze_button = st.button("Analyze Image", use_container_width=True, type="primary", disabled=not uploaded_file)
        if not uploaded_file:
            st.info("Please upload an image to enable the 'Analyze' button.")

    # Store results in session state to persist them across reruns
    if analyze_button:
        with st.spinner("Analyzing... This may take a moment."):
            image = Image.open(uploaded_file).convert("RGB")
            orig_img, cam_img, predictions = process_and_predict(image)
            
            st.session_state['results'] = {
                'orig_img': orig_img,
                'cam_img': cam_img,
                'predictions': predictions,
                'filename': uploaded_file.name
            }

    # Display results if they exist in the session state
    if 'results' in st.session_state:
        st.markdown("---")
        st.markdown("### 3. Review Results")
        
        results = st.session_state['results']
        predictions = results['predictions']
        
        # Display images side-by-side with info icons
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            display_image_with_info_icon(results['orig_img'], "Original Image", ORIGINAL_IMG_INFO)
        with img_col2:
            display_image_with_info_icon(results['cam_img'], "Grad-CAM Overlay", GRADCAM_IMG_INFO)
        
        st.markdown("#### Top 5 Predictions")
        
        # Create columns for prediction display
        pred_cols = st.columns(5)
        for i, (cls, conf) in enumerate(predictions):
            with pred_cols[i]:
                bg_color = get_gradient_color(float(conf))
                st.markdown(
                    f"""
                    <div class="prediction-container" style="background-color: {bg_color};">
                        <div class="prediction-label" title="{cls}">{cls}</div>
                        <div class="prediction-confidence">{conf:.1%}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.progress(float(conf), text="") # Progress bar without text

        # Disease Information and Symptoms
        st.markdown("---")
        st.markdown("### 4. Disease Information & Symptoms")
        
        # Get information for the top predicted condition
        top_prediction_label = predictions[0][0]
        disease_info = get_disease_info(top_prediction_label)
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("#### 🔍 **Disease Information**")
            st.markdown(f"**Condition:** {top_prediction_label}")
            st.markdown(f"**Confidence:** {predictions[0][1]:.1%}")
            st.markdown("**Description:**")
            st.info(disease_info['description'])
            
        with info_col2:
            st.markdown("#### ⚠️ **Common Symptoms**")
            st.markdown("**Typical symptoms associated with this condition:**")
            for symptom in disease_info['symptoms']:
                st.markdown(f"• {symptom}")
            
            st.warning("⚕️ **Medical Disclaimer:** This analysis is for informational purposes only. Please consult with a qualified ophthalmologist for proper diagnosis and treatment.")
        
        # Keep the text report download
        st.markdown("---")
        st.markdown("### 5. Download Report")
        report_data = create_report_text(predictions, results['filename'])
        st.download_button(
            label="📄 Download Analysis Report (.txt)",
            data=report_data,
            file_name=f"report_{results['filename']}.txt",
            mime="text/plain",
            use_container_width=True
        )

with tab2:
    # GenAI Assistant placeholder
    st.markdown("""
        <div style="text-align:center; padding:2rem; background-color:#f8f9fa; border-radius:0.5rem;">
            <h3>GenAI Assistant</h3>
            <p style="color:#6c757d;"><b>Coming Soon:</b> An AI assistant to generate detailed reports, suggest potential follow-ups, and provide reasoning based on the analysis.</p>
        </div>
    """, unsafe_allow_html=True)
