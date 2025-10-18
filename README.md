# 🎨 AI Canvas - MNIST Digit Classifier

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-00C7B7?style=for-the-badge&logo=ai&logoColor=white)

A modern, interactive web application that uses deep learning to recognize handwritten digits in real-time. Built with Streamlit and TensorFlow, this project demonstrates the power of AI in a beautiful, user-friendly interface.

## 🚀 Live Demo

**Experience the AI magic here:** [AI Canvas Live Demo](https://josephkamau32-ai-tools-app-60eipd.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://josephkamau32-ai-tools-app-60eipd.streamlit.app/)

## 📸 App Preview

![AI Canvas Interface](https://via.placeholder.com/800x400/667eea/ffffff?text=AI+Canvas+-+Draw+and+Watch+AI+Recognize+Digits)

## ✨ Features

### 🎨 Interactive Drawing Canvas
- Draw digits freely with an intuitive canvas interface
- Real-time digit recognition with instant predictions
- Beautiful, modern UI with gradient designs and smooth animations

### 🤖 Advanced AI Capabilities
- **CNN Model**: Convolutional Neural Network trained on MNIST dataset
- **High Accuracy**: Achieves over 98% accuracy on test data
- **Confidence Scoring**: Visual confidence breakdown for each prediction
- **Smart Preprocessing**: Automatic image enhancement for better recognition

### 📊 Comprehensive Analytics
- **Real-time Confidence Metrics**: See how confident the AI is in its prediction
- **Probability Distribution**: Visual breakdown of all possible digits
- **Performance Statistics**: Model accuracy and prediction history
- **Comparison Tools**: Side-by-side view of original and processed images

### 🎯 User Experience
- **Mobile Responsive**: Works seamlessly on all devices
- **No Setup Required**: Ready-to-use web application
- **Example Demonstrations**: Learn with pre-loaded digit examples
- **Helpful Tips**: Guidance for optimal drawing techniques

## 🏗️ Project Structure

```
ai-tools-assignment/
├── 📁 deployment/
│   ├── app.py                 # Main Streamlit application
│   ├── mnist_model.h5         # Trained TensorFlow model
│   └── model_info.json        # Model metadata and accuracy
├── 📁 part1_theory/           # Theoretical understanding answers
├── 📁 part2_practical/        # Practical implementation code
│   ├── task1_sklearn/         # Iris classification with Scikit-learn
│   ├── task2_tensorflow/      # MNIST classification with TensorFlow
│   └── task3_spacy/           # NLP analysis with spaCy
├── 📁 part3_ethics/           # Ethical considerations and analysis
├── requirements.txt           # Project dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Web application framework
- **TensorFlow** - Deep learning model training and inference
- **Keras** - High-level neural networks API

### Machine Learning & AI
- **Convolutional Neural Networks** - For image recognition
- **Scikit-learn** - Traditional machine learning algorithms
- **spaCy** - Natural language processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Image Processing
- **Pillow (PIL)** - Image manipulation and preprocessing
- **OpenCV** - Computer vision tasks

### Data Visualization
- **Matplotlib** - Static, animated, and interactive visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive plotting library

### Deployment & Infrastructure
- **Streamlit Cloud** - Application hosting and deployment
- **GitHub** - Version control and code repository
- **Google Colab** - Development and experimentation environment

## 📋 AI Assignment Overview

This project was developed as part of the "Mastering the AI Toolkit" assignment, covering:

### Part 1: Theoretical Understanding
- **TensorFlow vs PyTorch** comparative analysis
- **Jupyter Notebooks** use cases in AI development
- **spaCy vs basic string operations** in NLP
- **Scikit-learn vs TensorFlow** framework comparison

### Part 2: Practical Implementation
- **Task 1**: Iris classification with Decision Trees (Scikit-learn)
- **Task 2**: MNIST digit classification with CNN (TensorFlow)
- **Task 3**: Named Entity Recognition and sentiment analysis (spaCy)

### Part 3: Ethics & Optimization
- **Bias analysis** in MNIST and Amazon Reviews models
- **Fairness indicators** and mitigation strategies
- **Model optimization** and troubleshooting techniques

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/josephkamau32/ai_tools.git
   cd ai_tools
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run deployment/app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Using Docker (Alternative)

```bash
# Build the Docker image
docker build -t ai-canvas .

# Run the container
docker run -p 8501:8501 ai-canvas
```

## 🎮 How to Use

1. **Access the App**: Visit the [live demo](https://josephkamau32-ai-tools-app-60eipd.streamlit.app/) or run locally
2. **Draw a Digit**: Use the canvas to draw a digit (0-9)
3. **Get Prediction**: Click "Predict" to see the AI's analysis
4. **View Results**: Examine the confidence scores and probability distribution
5. **Try Examples**: Use the "Example" button to test with pre-loaded digits

### Tips for Best Results
- Draw digits large and centered
- Use thick, clear lines
- Ensure high contrast (white on black)
- Avoid background noise and artifacts

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.5% |
| **Precision** | 98.7% |
| **Recall** | 98.4% |
| **F1-Score** | 98.5% |
| **Inference Speed** | < 100ms |

**Architecture**: CNN with 3 convolutional layers, 2 max-pooling layers, and dropout regularization

## 🌐 Deployment

This application is deployed on **Streamlit Cloud** with the following configuration:

- **Platform**: Streamlit Community Cloud
- **Region**: Global
- **Auto-deploy**: On every push to main branch
- **Resources**: Standard tier with 1GB RAM

### Deployment Status
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://josephkamau32-ai-tools-app-60eipd.streamlit.app/)

## 🔧 Development

### Project Setup for Developers

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and test locally**
4. **Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Test model inference
python test_model.py

# Test Streamlit app locally
streamlit run deployment/app.py
```

## 📈 Future Enhancements

- [ ] **Multi-digit recognition** for sequences of numbers
- [ ] **Model retraining interface** for continuous learning
- [ ] **Export functionality** for saving predictions
- [ ] **User accounts** for prediction history
- [ ] **Advanced analytics** with prediction trends
- [ ] **Mobile app version** with camera integration
- [ ] **Custom model training** with user-uploaded datasets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- UI/UX improvements
- Model performance optimization
- Additional features and functionality
- Documentation enhancements
- Bug fixes and testing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Streamlit Team**: For the amazing web framework
- **TensorFlow Community**: For comprehensive deep learning tools
- **Google Colab**: For providing free GPU resources during development

## 📞 Contact & Support

**Developer**: Joseph Kamau  
**Email**: info@telivus.co.ke 
**GitHub**: [@josephkamau32](https://github.com/josephkamau32)  

### Support the Project

If you find this project helpful, please consider:
- ⭐ **Starring the repository**
- 🐛 **Reporting issues**
- 💡 **Suggesting new features**
- 🔄 **Sharing with others**

---

<div align="center">

**Made with ❤️ using Streamlit and TensorFlow**

[![Open in GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/josephkamau32/ai_tools)
[![Try the App](https://img.shields.io/badge/Try_the_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://josephkamau32-ai-tools-app-60eipd.streamlit.app/)

</div>