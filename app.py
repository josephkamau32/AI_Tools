import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image, ImageOps
import io
import time

# Page configuration
st.set_page_config(
    page_title="AI Canvas - MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        height: 8px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .drawing-canvas {
        border: 3px dashed #667eea;
        border-radius: 15px;
        cursor: crosshair;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .accuracy-badge {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🎨 AI Canvas</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Draw a digit and watch our AI magically recognize it!</h2>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model"""
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess the drawn image for model prediction using Pillow only"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 using high-quality resampling
    img_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    # Invert colors (MNIST is white on black)
    img_array = 255 - img_array
    
    # Normalize to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model (batch_size, height, width, channels)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def enhance_image_contrast(image):
    """Enhance image contrast using Pillow for better prediction"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    image = ImageOps.autocontrast(image, cutoff=2)
    
    return image

def create_confidence_chart(probabilities):
    """Create a beautiful confidence chart"""
    chart_data = {
        'Digit': [str(i) for i in range(10)],
        'Confidence': probabilities
    }
    return chart_data

def main():
    # Load model
    model, model_info = load_model()
    
    if model is None:
        st.error("❌ Model failed to load. Please check if mnist_model.h5 exists.")
        return
    
    # Display model accuracy
    accuracy = model_info.get('accuracy', 0)
    st.markdown(f'<div style="text-align: center;"><span class="accuracy-badge">Model Accuracy: {accuracy:.2%}</span></div>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎨 Drawing Canvas")
        st.markdown("Draw a digit (0-9) in the box below:")
        
        # Try to import streamlit-drawable-canvas, fallback to file uploader
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            canvas_available = True
            
        except ImportError:
            st.warning("⚠️ streamlit-drawable-canvas not installed. Using file uploader instead.")
            canvas_available = False
            canvas_result = None
        
        # Alternative file uploader
        if not canvas_available:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("### 📁 Upload Image")
            uploaded_file = st.file_uploader("Upload a digit image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=200)
                    canvas_result = type('obj', (object,), {'image_data': np.array(image)})
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Control buttons
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            predict_clicked = st.button("🔍 Predict", use_container_width=True)
        with col1_2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
        with col1_3:
            example_clicked = st.button("💡 Example", use_container_width=True)
        
        # Handle predictions
        if predict_clicked:
            if canvas_result and canvas_result.image_data is not None:
                process_prediction(canvas_result.image_data, model)
            else:
                st.warning("Please draw a digit or upload an image first!")
        
        # Handle example
        if example_clicked:
            show_example_prediction(model)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        # Initialize session state for predictions
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
            st.session_state.last_confidence = None
            st.session_state.last_probabilities = None
            st.session_state.last_image = None
        
        # Display last prediction
        if st.session_state.last_prediction is not None:
            display_prediction_results()

def process_prediction(image_data, model):
    """Process the drawn image and make prediction"""
    try:
        # Convert canvas image to PIL Image
        if isinstance(image_data, np.ndarray):
            img_pil = Image.fromarray((image_data[:, :, :3]).astype('uint8'))
        else:
            img_pil = image_data
        
        # Store the original image for display
        st.session_state.last_image = img_pil
        
        # Enhance contrast for better prediction
        enhanced_image = enhance_image_contrast(img_pil)
        
        # Preprocess image
        processed_image = preprocess_image(enhanced_image)
        
        # Make prediction
        with st.spinner('🔄 AI is analyzing your drawing...'):
            time.sleep(1)  # Add dramatic effect
            predictions = model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        probabilities = predictions[0].tolist()
        
        # Store in session state
        st.session_state.last_prediction = predicted_digit
        st.session_state.last_confidence = confidence
        st.session_state.last_probabilities = probabilities
        
        # Success animation
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

def display_prediction_results():
    """Display the prediction results beautifully"""
    pred = st.session_state.last_prediction
    conf = st.session_state.last_confidence
    probs = st.session_state.last_probabilities
    
    # Show the processed image
    if st.session_state.last_image is not None:
        st.markdown("### 🖼️ Your Drawing")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Display original image
            st.image(st.session_state.last_image, caption="Original", width=150)
        
        with col2:
            # Display processed version
            processed_img = preprocess_image(st.session_state.last_image)
            processed_display = (processed_img[0, :, :, 0] * 255).astype('uint8')
            processed_pil = Image.fromarray(processed_display)
            st.image(processed_pil, caption="Processed for AI", width=150)
    
    # Main prediction card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    # Animated result
    if conf > 0.8:
        st.markdown(f"## 🎉 Predicted Digit: **{pred}**")
    elif conf > 0.6:
        st.markdown(f"## 🔢 Predicted Digit: **{pred}**")
    else:
        st.markdown(f"## 🤔 Predicted Digit: **{pred}**")
    
    # Confidence meter with color coding
    if conf > 0.9:
        confidence_color = "🟢"
    elif conf > 0.7:
        confidence_color = "🟡"
    else:
        confidence_color = "🔴"
        
    st.markdown(f"### {confidence_color} Confidence: **{conf:.2%}**")
    st.progress(float(conf))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence breakdown
    st.markdown("### 📈 Confidence Breakdown")
    
    # Create columns for digit probabilities
    cols = st.columns(5)
    for i in range(10):
        with cols[i % 5]:
            prob = probs[i]
            if i == pred:
                emoji = "🎯"
                delta_color = "normal"
            else:
                emoji = "⚪"
                delta_color = "off"
            
            st.metric(
                label=f"{emoji} {i}", 
                value=f"{prob:.2%}",
                delta=None,
                delta_color=delta_color
            )
    
    # Visual confidence chart
    st.markdown("### 📊 Confidence Visualization")
    chart_data = create_confidence_chart(probs)
    st.bar_chart(chart_data, x='Digit', y='Confidence', use_container_width=True)
    
    # Prediction tips
    with st.expander("💡 Prediction Tips"):
        st.write("""
        - **Draw clearly**: Make digits large and centered
        - **Use thick lines**: Better for the AI to recognize
        - **Avoid noise**: Keep the background clean
        - **High contrast**: White on black works best
        - **Confidence > 80%**: Usually indicates good recognition
        """)

def show_example_prediction(model):
    """Show an example prediction with a sample digit"""
    try:
        # Generate a sample digit (using a random test sample)
        from tensorflow.keras.datasets import mnist
        (_, _), (X_test, y_test) = mnist.load_data()
        
        # Find a clear example (not too faint)
        good_examples = []
        for i in range(len(X_test)):
            if np.max(X_test[i]) > 200:  # Look for clear, dark digits
                good_examples.append(i)
            if len(good_examples) > 50:
                break
        
        if good_examples:
            sample_idx = np.random.choice(good_examples)
            sample_digit = X_test[sample_idx]
            true_label = y_test[sample_idx]
            
            # Convert to PIL Image
            sample_img = Image.fromarray(sample_digit)
            
            # Store for display
            st.session_state.last_image = sample_img
            
            # Process and predict
            processed_image = preprocess_image(sample_img)
            predictions = model.predict(processed_image, verbose=0)
            
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            probabilities = predictions[0].tolist()
            
            # Store results
            st.session_state.last_prediction = predicted_digit
            st.session_state.last_confidence = confidence
            st.session_state.last_probabilities = probabilities
            
            # Show the example image
            st.success(f"✅ Loaded example digit: {true_label}")
            
        else:
            st.error("Could not find a good example digit")
            
    except Exception as e:
        st.error(f"Error loading example: {e}")

# Handle streamlit-drawable-canvas import gracefully
try:
    from streamlit_drawable_canvas import st_canvas
    canvas_imported = True
except ImportError:
    canvas_imported = False

if __name__ == "__main__":
    main()