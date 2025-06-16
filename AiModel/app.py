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
import os
import requests

# RAG Chatbot imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

# ==============================================================================
# Page Configuration (must be the first Streamlit command)
# ==============================================================================
st.set_page_config(
    page_title="Fundus Classifier with Grad-CAM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Enhanced CSS for Grid-based Layout
# ==============================================================================
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main > div {
            max-width: 1400px; 
            padding-top: 1rem;
            font-family: 'Inter', sans-serif;
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            margin-top: 0; 
            margin-bottom: 0.75rem; 
            color: #1a365d;
            font-weight: 600;
        }
        
        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }
        h4 { font-size: 1.25rem; }
        
        /* Container Styles */
        div.block-container {
            padding-top: 1rem; 
            padding-bottom: 1rem;
        }
        
        /* Grid Container */
        .grid-container {
            display: grid;
            gap: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Card Styles */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Upload Section */
        .upload-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Analysis Grid */
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }
        
        /* Prediction Grid - Single Row */
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }

        @media (max-width: 1200px) {
            .prediction-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (max-width: 768px) {
            .prediction-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 480px) {
            .prediction-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .prediction-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .prediction-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4299e1, #3182ce);
        }
        
        .prediction-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .prediction-label {
            font-weight: 600;
            font-size: 0.95rem;
            color: #2d3748;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }
        
        .prediction-confidence {
            font-size: 1.1rem;
            font-weight: 700;
            color: #3182ce;
            margin-bottom: 0.5rem;
        }
        
        .prediction-rank {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #4299e1;
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        /* Info Section Grid */
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .info-card {
            background: #f7fafc;
            border-left: 4px solid #4299e1;
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        .info-card h4 {
            color: #2b6cb0;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Image Container */
        .image-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .stImage > img {
            border-radius: 12px;
            border: none;
        }
        
        /* Button Styles */
        div.stButton > button {
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(66, 153, 225, 0.3);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #b442e1, #b442e1);
            border-radius: 4px;
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-high { background: #fed7d7; color: #c53030; }
        .status-medium { background: #feebc8; color: #dd6b20; }
        .status-low { background: #c6f6d5; color: #38a169; }
        
        /* Chat Styles */
        .chat-container {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .assistant-response {
            background: #f7fafc;
            border-left: 4px solid #4299e1;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .example-questions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .example-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .example-card:hover {
            background: #e9ecef;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .analysis-grid, .info-grid {
                grid-template-columns: 1fr;
            }
            
            .prediction-grid {
                grid-template-columns: 1fr 1fr;
            }
            
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
        }
        
        /* Section Dividers */
        .section-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #4299e1, transparent);
            margin: 3rem 0 2rem 0;
        }
        
        /* Disclaimer Box */
        .disclaimer-box {
            background: #fff5f5;
            border: 1px solid #feb2b2;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .disclaimer-box .disclaimer-icon {
            color: #e53e3e;
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Configuration and Constants
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 299
# IMPORTANT: Replace this with the actual path to your model file
MODEL_PATH = r'D:/Kaif/Hackathon25/CDAC/AiModel/Model 2/best_fundus_efficientnetb3.pth' 
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

# ==== RAG CHATBOT CONFIG ====
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # or set manually
GROQ_API_KEY = ''  # or set manually
MODEL_ID = "llama3-70b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

EMBED_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./chromadb"

# Disease Information Database
DISEASE_INFO = {
    'Normal': {
        'description': 'The fundus appears healthy with no signs of pathological changes.',
        'symptoms': ['No symptoms', 'Regular vision', 'No visual disturbances'],
        'severity': 'low'
    },
    'Tessellated fundus': {
        'description': 'A benign condition where the choroidal blood vessels are visible through a thin retinal pigment epithelium.',
        'symptoms': ['Usually asymptomatic', 'May have mild visual changes', 'No pain or discomfort'],
        'severity': 'low'
    },
    'Large optic cup': {
        'description': 'An enlarged optic cup which may indicate glaucoma or other optic nerve conditions.',
        'symptoms': ['Gradual vision loss', 'Peripheral vision loss', 'Difficulty seeing in low light'],
        'severity': 'medium'
    },
    'DR1': {
        'description': 'Mild non-proliferative diabetic retinopathy - early stage of diabetic eye disease.',
        'symptoms': ['Often no symptoms', 'Occasional blurred vision', 'Difficulty focusing'],
        'severity': 'medium'
    },
    'DR2': {
        'description': 'Moderate non-proliferative diabetic retinopathy with more advanced retinal changes.',
        'symptoms': ['Blurred vision', 'Dark spots in vision', 'Difficulty reading'],
        'severity': 'medium'
    },
    'DR3': {
        'description': 'Severe non-proliferative diabetic retinopathy requiring immediate attention.',
        'symptoms': ['Significant vision loss', 'Floaters', 'Dark areas in vision', 'Difficulty with night vision'],
        'severity': 'high'
    },
    'Possible glaucoma': {
        'description': 'Signs suggesting glaucoma, a condition that damages the optic nerve.',
        'symptoms': ['Gradual peripheral vision loss', 'Eye pain', 'Halos around lights', 'Headaches'],
        'severity': 'high'
    },
    'Optic atrophy': {
        'description': 'Damage or deterioration of the optic nerve.',
        'symptoms': ['Vision loss', 'Reduced color vision', 'Decreased visual field', 'Poor night vision'],
        'severity': 'high'
    },
    'Severe hypertensive retinopathy': {
        'description': 'Severe damage to retinal blood vessels due to high blood pressure.',
        'symptoms': ['Sudden vision loss', 'Headaches', 'Double vision', 'Dim vision'],
        'severity': 'high'
    },
    'Disc swelling and elevation': {
        'description': 'Swelling of the optic disc, which may indicate increased intracranial pressure.',
        'symptoms': ['Headaches', 'Vision changes', 'Nausea', 'Double vision'],
        'severity': 'high'
    },
    'Maculopathy': {
        'description': 'Disease affecting the macula, the central part of the retina.',
        'symptoms': ['Central vision loss', 'Distorted vision', 'Difficulty reading', 'Problems recognizing faces'],
        'severity': 'high'
    },
    'Pathological myopia': {
        'description': 'Severe nearsightedness that can lead to retinal complications.',
        'symptoms': ['Severe nearsightedness', 'Floaters', 'Flashing lights', 'Distorted vision'],
        'severity': 'medium'
    }
}

# ==== Vector Store Setup (runs once) ====
@st.cache_resource
def load_retriever():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error loading retriever: {e}")
        return None

# ==== System Prompt ====
SYSTEM_PROMPT = """
You are a medical assistant specialized in ophthalmology and eye-related diseases.

When answering user queries:
- Use simple, medically accurate language.
- Explain the condition, symptoms, diagnosis methods, and available treatments.
- Suggest when a patient should see a doctor.
- Provide information in bullet points and summaries where needed.
- Always emphasize that this is for informational purposes only and not a substitute for professional medical advice.

If multiple diseases are relevant, include them all.
"""

def get_disease_info(label):
    """Get disease information and symptoms for a given label"""
    clean_label = label.replace('_', ' ').strip()
    
    if clean_label in DISEASE_INFO:
        return DISEASE_INFO[clean_label]
    
    for key in DISEASE_INFO.keys():
        if key.lower() in clean_label.lower() or clean_label.lower() in key.lower():
            return DISEASE_INFO[key]
    
    return {
        'description': 'This condition requires further evaluation by an ophthalmologist for proper diagnosis and treatment.',
        'symptoms': ['Consult with an eye care professional', 'Regular eye examinations recommended', 'Monitor any vision changes'],
        'severity': 'medium'
    }

# Image transformations
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
def get_severity_class(severity):
    """Get CSS class for severity indicator"""
    severity_map = {
        'low': 'status-low',
        'medium': 'status-medium',
        'high': 'status-high'
    }
    return severity_map.get(severity, 'status-medium')

def get_severity_icon(severity):
    """Get icon for severity level"""
    icon_map = {
        'low': '✅',
        'medium': '⚠️',
        'high': '🚨'
    }
    return icon_map.get(severity, '⚠️')

def process_and_predict(image: Image.Image):
    """Processes an image, runs prediction, and generates Grad-CAM overlay."""
    model = load_model()
    cam_extractor = GradCAM(model, target_layer="conv_head")
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    scores = model(input_tensor)
    probs = torch.sigmoid(scores).cpu().detach().numpy()[0]
    
    top5_idx = probs.argsort()[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_labels = [CLASS_NAMES[i].split('.')[-1].replace('_', ' ') for i in top5_idx]
    
    predicted_idx = int(top5_idx[0])
    cam_map = cam_extractor(class_idx=predicted_idx, scores=scores)[0].cpu().squeeze()
    
    display_img = input_tensor.squeeze().cpu()
    display_img = torch.clamp((display_img * 0.5) + 0.5, 0, 1)
    img_pil = to_pil_image(display_img)
    
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
        report += f"- {label}: {conf:.2%}\n"
    
    report += "\nDisclaimer: This is an automated analysis and not a substitute for professional medical advice."
    return report

# ==== RAG Query Handler ====
def generate_answer(user_query: str) -> str:
    retriever = load_retriever()
    if retriever is None:
        return "Sorry, the knowledge base is not available at the moment. Please try again later."
    
    if not GROQ_API_KEY:
        return "Please set your GROQ_API_KEY environment variable to use the AI assistant."
    
    try:
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(user_query)])

        enhanced_prompt = f"{SYSTEM_PROMPT}\n\n---\n\nRelevant Medical Info:\n{context}\n\n---\n\nUser Query:\n{user_query}"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.6,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.json()}")

        result = response.json()
        return result['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ==============================================================================
# Main User Interface
# ==============================================================================

# Header Section
st.markdown("""
    <div class="upload-section">
        <h1>🔬 Advanced Fundus Image Classifier</h1>
        <p style="font-size: 1.2rem; margin-bottom: 0;">AI-Powered Retinal Disease Detection with Visual Explanations</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["**📊 Analysis Dashboard**", "**🤖 GenAI Assistant**"])

with tab1:
    # Upload and Analysis Section
    st.markdown("## 📤 Image Upload & Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("### Upload Fundus Image")
        uploaded_file = st.file_uploader(
            "Choose a fundus image (JPG, PNG)...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear fundus photograph for analysis"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col2:
        st.markdown("### ")  # Spacer
        st.markdown("<div style='text-align: center; padding: 2rem 0;'>➡️</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Run Analysis")
        analyze_button = st.button(
            "🔍 Analyze Image", 
            use_container_width=True, 
            type="primary", 
            disabled=not uploaded_file,
            help="Click to start AI analysis of the uploaded image"
        )
        
        if not uploaded_file:
            st.info("📋 Please upload an image to enable analysis")

    # Process Analysis
    if analyze_button:
        with st.spinner("🔄 Analyzing image... This may take a moment."):
            image = Image.open(uploaded_file).convert("RGB")
            orig_img, cam_img, predictions = process_and_predict(image)
            
            st.session_state['results'] = {
                'orig_img': orig_img,
                'cam_img': cam_img,
                'predictions': predictions,
                'filename': uploaded_file.name
            }

    # Display Results
    if 'results' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        results = st.session_state['results']
        predictions = results['predictions']
        
        # Image Analysis Grid
        st.markdown("## 🖼️ Visual Analysis Results")
        
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("###  Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(results['orig_img'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("💡 Original fundus photograph showing retinal structures")
        
        with img_col2:
            st.markdown("###  AI Focus Areas (Grad-CAM)")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(results['cam_img'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("🔍 Red areas show regions the AI focused on for diagnosis")

        # Predictions Grid
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📈 Prediction Results")

        # Create all 5 prediction cards in a single row
        cols = st.columns(5)

        for i, (cls, conf) in enumerate(predictions):
            with cols[i]:
                rank = i + 1
                confidence_pct = conf * 100
                
                # Determine confidence level for styling
                if confidence_pct >= 70:
                    conf_class = "high"
                elif confidence_pct >= 40:
                    conf_class = "medium"
                else:
                    conf_class = "low"
                
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-rank">{rank}</div>
                        <div class="prediction-label" title="{cls}">{cls}</div>
                        <div class="prediction-confidence">{confidence_pct:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add progress bar
                st.progress(float(conf))

        # Disease Information Grid
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 🏥 Medical Information")
        
        top_prediction_label = predictions[0][0]
        disease_info = get_disease_info(top_prediction_label)
        severity = disease_info.get('severity', 'medium')
        severity_icon = get_severity_icon(severity)
        severity_class = get_severity_class(severity)
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"""
                <div class="info-card">
                    <h4> Primary Diagnosis</h4>
                    <div style="margin-bottom: 1rem;">
                        <strong>Condition:</strong> {top_prediction_label}<br>
                        <strong>Confidence:</strong> {predictions[0][1]:.1%}<br>
                        <strong>Severity:</strong> 
                        <span class="status-indicator {severity_class}">
                            {severity_icon} {severity.title()}
                        </span>
                    </div>
                    <div>
                        <strong>Description:</strong><br>
                        {disease_info['description']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown(f"""
                <div class="info-card">
                    <h4> Associated Symptoms</h4>
                    <div>
                        <strong>Common symptoms may include:</strong>
                        <ul style="margin-top: 0.5rem;">
            """, unsafe_allow_html=True)
            
            for symptom in disease_info['symptoms']:
                st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div></div>", unsafe_allow_html=True)

        # Download Section
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📄 Export Results")
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            report_data = create_report_text(predictions, results['filename'])
            st.download_button(
                label="📄 Download Report (.txt)",
                data=report_data,
                file_name=f"fundus_analysis_{results['filename']}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with download_col2:
            st.button(
                "📧 Email Report",
                use_container_width=True,
                help="Feature coming soon",
                disabled=True
            )
        
        with download_col3:
            st.button(
                "🖨️ Print Report",
                use_container_width=True,
                help="Feature coming soon",
                disabled=True
            )

with tab2:
    # RAG Chatbot Tab
    st.markdown("""
        <div class="chat-header">
            <h2>🧠 Ophthalmology GenAI Assistant</h2>
            <p style="font-size: 1.1rem; margin-bottom: 0;">
                Ask questions about eye diseases, symptoms, treatments, and diagnosis using our AI-powered medical knowledge base.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    
    example_questions = [
        "What are the symptoms of diabetic retinopathy?",
        "How is glaucoma diagnosed?",
        "What causes macular degeneration?",
        "What are the treatment options for retinal detachment?",
        "How can I prevent eye diseases?",
        "What is the difference between DR1, DR2, and DR3?"
    ]
    
    st.markdown('<div class="example-questions">', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                st.session_state['selected_question'] = question
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("### 🗣️ Ask Your Question")
    
    with st.form("eye_chat_form"):
        # Use selected question if available
        default_question = st.session_state.get('selected_question', '')
        user_question = st.text_area(
            "👁️ What would you like to ask about eye diseases?",
            value=default_question,
            height=100,
            placeholder="e.g., What are the early signs of glaucoma?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.form_submit_button("🚀 Ask Assistant", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
        if clear:
            st.session_state['selected_question'] = ''
            st.rerun()

        if submit and user_question.strip():
            with st.spinner("🔄 Analyzing with medical knowledge..."):
                try:
                    answer = generate_answer(user_question)
                    
                    st.markdown('<div class="assistant-response">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Assistant's Answer:")
                    st.markdown(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clear the selected question after answering
                    if 'selected_question' in st.session_state:
                        del st.session_state['selected_question']
                        
                except Exception as e:
                    st.error(f"🚫 Error: {e}")
    
    # Additional information
    st.markdown("---")
    st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
            <h4>ℹ️ About the AI Assistant</h4>
            <p>This AI assistant uses a comprehensive medical knowledge base focused on ophthalmology and eye diseases. 
            It can help you understand:</p>
            <ul>
                <li>🔍 Disease symptoms and characteristics</li>
                <li>🏥 Diagnostic procedures and methods</li>
                <li>💊 Treatment options and recommendations</li>
                <li>🛡️ Prevention strategies</li>
                <li>📚 General eye health information</li>
            </ul>
            <p><strong>Important:</strong> This information is for educational purposes only and should not replace professional medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; 
                border-top: 1px solid #e2e8f0; color: #718096;">
        <p>Powered by AI • Built with Streamlit • For Educational Purposes</p>
    </div>
""", unsafe_allow_html=True)