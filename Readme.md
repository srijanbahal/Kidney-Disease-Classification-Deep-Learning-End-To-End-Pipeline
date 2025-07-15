# Kidney Disease Classification Deep Learning Project

A comprehensive end-to-end deep learning project for kidney disease classification using computer vision techniques, MLOps practices, and cloud deployment strategies.

## 🎯 Project Overview

This project implements a robust deep learning pipeline for classifying kidney diseases from medical images. It demonstrates production-grade MLOps practices including experiment tracking, model versioning, containerization, and automated deployment on AWS cloud infrastructure.

## 🚀 Key Features

- **Deep Learning Classification**: CNN-based models for kidney disease detection
- **MLOps Integration**: Complete MLflow and DVC pipeline implementation
- **Cloud Deployment**: AWS EC2 and ECR integration for scalable deployment
- **Experiment Tracking**: Comprehensive logging and model versioning
- **CI/CD Pipeline**: Automated deployment with GitHub Actions
- **Containerization**: Docker-based application packaging

## 🏗️ Project Architecture

### ML Pipeline Components
1. **Data Ingestion**: Automated data collection and preprocessing
2. **Data Validation**: Quality checks and schema validation
3. **Model Training**: Deep learning model training with hyperparameter tuning
4. **Model Evaluation**: Performance metrics and validation
5. **Model Deployment**: Production-ready model serving

### Technology Stack
- **Framework**: TensorFlow/Keras for deep learning
- **Experiment Tracking**: MLflow for model lifecycle management
- **Pipeline Orchestration**: DVC for reproducible ML pipelines
- **Containerization**: Docker for application packaging
- **Cloud Platform**: AWS (EC2, ECR) for deployment
- **CI/CD**: GitHub Actions for automated workflows

## 📁 Project Structure

```
├── src/
│   ├── cnnClassifier/
│   │   ├── components/          # Core ML components
│   │   ├── config/             # Configuration management
│   │   ├── constants/          # Project constants
│   │   ├── entity/             # Data entities and schemas
│   │   ├── pipeline/           # ML pipeline stages
│   │   └── utils/              # Utility functions
├── config/
│   ├── config.yaml            # Main configuration
│   ├── params.yaml            # Model parameters
│   └── secrets.yaml           # Sensitive configurations
├── artifacts/                 # Generated artifacts and models
├── logs/                     # Application logs
├── templates/                # Web application templates
├── static/                   # Static files for web app
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── main.py                  # Training pipeline entry point
├── app.py                   # Web application
├── dvc.yaml                 # DVC pipeline configuration
└── Dockerfile              # Container configuration
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Conda or virtual environment
- Docker (for containerization)
- AWS CLI (for cloud deployment)

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project
   cd Kidney-Disease-Classification-Deep-Learning-Project
   ```

2. **Create Virtual Environment**
   ```bash
   conda create -n cnncls python=3.8 -y
   conda activate cnncls
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the Project**
   - Update `config/config.yaml` with your configurations
   - Update `config/params.yaml` with model parameters
   - Update `config/secrets.yaml` with sensitive information (optional)

## 🚀 Usage

### Training the Model

1. **Run the Training Pipeline**
   ```bash
   python main.py
   ```

2. **Track Experiments with DVC**
   ```bash
   dvc init
   dvc repro
   dvc dag
   ```

### Web Application

1. **Launch the Application**
   ```bash
   python app.py
   ```

2. **Access the Interface**
   - Open your browser and navigate to `http://localhost:8080`
   - Upload kidney images for classification

### MLflow Tracking

1. **Start MLflow UI**
   ```bash
   mlflow ui
   ```

2. **Remote Tracking (Production)**
   ```bash
   export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow
   export MLFLOW_TRACKING_USERNAME=entbappy  
   export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0
   ```

## ☁️ Cloud Deployment

### AWS Setup Requirements

**Required AWS Services:**
- EC2 (Virtual Machine)
- ECR (Elastic Container Registry)

**Required IAM Policies:**
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

### Deployment Process

1. **Build Docker Image**
'   ```bash
   docker build -t kidney-classifier .
   ```

2. **Push to ECR**
   ```bash
   # Tag and push to your ECR repository
   docker tag kidney-classifier:latest 566373416292.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
   docker push 566373416292.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
   ```

3. **EC2 Instance Setup**
   ```bash
   # Update system
   sudo apt-get update -y
   sudo apt-get upgrade
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. **Deploy Application**
   ```bash
   # Pull and run the container
   docker pull 566373416292.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
   docker run -p 8080:8080 kidney-classifier:latest
   ```

### GitHub Actions CI/CD

Configure the following secrets in your GitHub repository:

```yaml
AWS_ACCESS_KEY_ID: <your-access-key>
AWS_SECRET_ACCESS_KEY: <your-secret-key>
AWS_REGION: us-east-1
AWS_ECR_LOGIN_URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME: kidney-classifier
```

## 📊 Model Performance

The project implements various deep learning architectures for optimal performance:
- Convolutional Neural Networks (CNNs)
- Transfer Learning approaches
- Model ensembling techniques

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🔄 MLOps Pipeline

### Experiment Tracking
- **MLflow**: Production-grade experiment tracking and model registry
- **DVC**: Lightweight pipeline orchestration and data versioning

### Model Lifecycle Management
1. **Development**: Local experimentation and model development
2. **Staging**: Model validation and testing
3. **Production**: Deployed model serving
4. **Monitoring**: Performance tracking and model drift detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Medical imaging dataset providers
- Open source community
- AWS for cloud infrastructure
- MLflow and DVC communities


⭐ **Star this repository if you find it helpful!** ⭐
