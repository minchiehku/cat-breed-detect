# cat-breed-detect


## catbreeddetect.com

This project is a web-based application designed to predict the breed of a cat from an uploaded image, utilizing a trained deep learning model for accurate classification. The model, developed using the **Cat Breeds Dataset** from Kaggle, is capable of classifying cat images into 67 distinct breeds. The development process includes initial training and model refinement on **Google Colab**, followed by further optimization and deployment on **AWS SageMaker**.

The application is containerized using **Docker** and **Docker Compose** for streamlined development and deployment. It is hosted on **AWS ECS (Fargate)**, with **ALB (Application Load Balancer)** ensuring scalability and **Route 53** managing domain routing and HTTPS support. This setup leverages AI technology and cloud services to create a robust, scalable platform for accurate and efficient cat breed recognition.



## catbreeddetect.com Architecture

![web](/aws_architecture/web_architec.jpg)


## Model Training Architecture

![model](/aws_architecture/training_architec.jpg)

---

## Dataset Processing

- **Dataset Source**: [Cat Breeds Dataset on Kaggle](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset)
- **Dataset Description**: Images of 67 cat breeds with attributes such as age, gender, size, coat, and breed.

### Data Preparation and Optimization

1. **Data Extraction and Structure Validation**
    - Downloaded and extracted the dataset from Kaggle.
    - Organized subdirectories to ensure each corresponds to a breed.
    - Resolved inconsistencies in file naming.
2. **Data Augmentation and Normalization**
    - Addressed imbalanced breed distribution by augmenting images (rotation, flipping, scaling, etc.).
    - Normalized pixel values to `[0, 1]`.
3. **Skipping Corrupted Images**
    - Implemented a safe data generator to skip invalid images during batch processing.
4. **Train-Test Split**
    - Split dataset into 80% training and 20% validation.
5. **Class Mapping and Storage**
    - Mapped breed names to numerical labels and saved as a JSON file.

---

## Machine Learning and Deep Learning

- **Framework**: TensorFlow/Keras
- **Transfer Learning**: Used `MobileNetV2` pretrained on ImageNet for feature extraction.
- **Image Preprocessing**:
    - Resized images to `(128, 128)`.
    - Augmented data with transformations to enhance generalizability.
- **Multi-class Classification**:
    - Applied **softmax** for probability distribution.
    - Predictions made using the class with the highest probability.

---

## Training on Google Colab

1. **Environment Setup**:
    - Installed Kaggle API to download the dataset.
    - Set up TensorFlow, Keras, and other necessary libraries.
    ```python    
    !pip install kaggle
    from google.colab import drive
    drive.mount('/content/drive')  # Mount Google Drive for saving models
    ``` 
   
2. **Data Preparation**:
    - Downloaded and preprocessed the dataset as described in the "Dataset Processing" section.
3. **Model Development**:
    - **Base Model**: `MobileNetV2` with pretrained weights from ImageNet.
    - **Additional Layers**:
        - `Flatten`
        - `Dense(256, activation='relu')`
        - `Dense(num_classes, activation='softmax')`
    
    **Training Code**:
    
    ```python    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.applications import MobileNetV2
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze pretrained layers
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(67, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)
    
    ```
    
4. **Saving Checkpoints**:
    - Saved the trained model checkpoint and class indices for later use.

    ```python    
    model.save('cat_breed_model_checkpoint.keras')
    with open('cat_breed_class_indices.json', 'w') as f:
        json.dump(class_indices, f)    
    ```
    

---

## Transition to AWS SageMaker

After completing initial training on Colab, the next step involved migrating the project to AWS for further training and deployment.

1. **Upload to AWS S3**:
    - Used `Boto3` to upload the dataset and Colab-generated checkpoints to an S3 bucket.
    
    **Key Code**:
    
    ```python    
    import boto3
    
    s3 = boto3.client('s3')
    bucket_name = 'my-cat-breeds'
    s3.upload_file('./cat_breed_model_checkpoint.keras', bucket_name, 'models/cat_breed_model_checkpoint.keras')
    s3.upload_file('./cat_breed_class_indices.json', bucket_name, 'models/cat_breed_class_indices.json')
    
    ```
    
2. **AWS SageMaker Training**:
    - Configured SageMaker to continue training using the uploaded checkpoint and dataset.
    
    **Steps**:
    
    - **Start SageMaker Notebook Instance**:
        - Selected a GPU-enabled instance type (e.g., `ml.p2.xlarge`).
    - **Load Data**:
        - Downloaded the dataset and checkpoint from S3.
    - **Resume Training**:
        - Reloaded the model and checkpoint, and resumed training for further optimization.
    
    **Key Code**:
    
    ```python    
    from sagemaker.tensorflow import TensorFlow
    
    estimator = TensorFlow(
        entry_point='train.py',
        role='YourSageMakerRole',
        instance_type='ml.p2.xlarge',
        framework_version='2.6.0',
        hyperparameters={
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001
        }
    )
    estimator.fit({'training': f's3://{bucket_name}/cat_breeds'})
    
    ```
    
3. **Model Deployment**:
    - Deployed the final trained model for real-time inference using SageMaker’s hosting services.
    
    **Key Code**:
    
    ```python    
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium'
    )
    response = predictor.predict(test_image_array)
    print(f"Predicted class: {response}")
    
    ```
    

---

## Results and Outputs

1. **Model Architecture**:
    - Base: `MobileNetV2` with frozen weights.
    - Additional Layers:
        - `Flatten`
        - `Dense(256, activation='relu')`
        - `Dense(num_classes, activation='softmax')`
2. **Final Outputs**:
    - Checkpoints and models saved in S3:
        - `cat_breed_model_checkpoint.keras`
        - `cat_breed_model.h5`

---

## Model Usage

1. **Image Classification**:
    - Use the trained model to classify images with confidence scores.
    
    ```python    
    def classify_image(img_path, model, class_indices_path):
        import json
        import numpy as np
        from tensorflow.keras.preprocessing import image
    
        with open(class_indices_path, 'r') as f:
            class_labels = json.load(f)
        class_labels = {v: k for k, v in class_labels.items()}
    
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]
        print(f"Predicted Breed: {class_labels[predicted_class]}, Confidence: {confidence:.2f}")
    
    ```
    
---

# catbreeddetect.com

## Features

- **Cat Breed Classification**: Utilizes a pre-trained TensorFlow model to recognize cat breeds and provide confidence levels.
- **Frontend and Backend Separation**:
    - **Frontend**: Interactive web interface for image upload and result display.
    - **Backend**: Flask-based API handling prediction requests and image uploads.
- **Containerization**: Uses Docker and Docker Compose to simplify development and deployment, ensuring consistency between local and cloud environments.
- **AWS Cloud Infrastructure**:
    - **Amazon ECS (Fargate)**: Manages containerized services with automatic scaling.
    - **ALB (Application Load Balancer)**: Distributes HTTP/HTTPS traffic and performs health checks.
    - **Route 53 and ACM**: Provides custom domain and SSL encryption.

---

## System Architecture

### Frontend

- **Technologies**: HTML, CSS, JavaScript, with Nginx serving static assets and proxying requests.
- **Protocol**: HTTP, container port 80.
- **ALB Health Check**: Root path `/`.

### Backend

- **Technologies**: Flask API handling image classification logic and model inference.
- **Protocol**: HTTP, container port 5000.
- **ALB Health Check**: Path `/predict`.

---

## Technologies Used

- **Frontend**:
    - **HTML/CSS/JavaScript**: For frontend page rendering.
    - **Nginx**: Acts as a reverse proxy and serves static assets.
- **Backend**:
    - **Flask**: Handles backend API requests.
    - **TensorFlow**: Performs deep learning model inference.
    - **Pillow**: Handles image upload and processing.
- **Deployment and Management**:
    - **Docker and Docker Compose**: For containerization and local testing.
    - **AWS ECS (Fargate)**: Manages running containerized applications.
    - **ALB and Route 53**: Provides load balancing and domain management.

---

## Deployment Instructions

### Prerequisites

- **AWS Account**: With access to ECS, ECR, Route 53, ACM, and ALB services.
- **Docker and Docker Compose**: Installed locally for building and testing images.
- **Custom Domain**: For HTTPS setup.

### Initial Setup

1. **Create ECR Repositories**:
    
    ```bash
    aws ecr create-repository --repository-name frontend-repo
    aws ecr create-repository --repository-name backend-repo
    
    ```
    
2. **Build Docker Images Locally**:
    
    ```bash
    docker build -t frontend ./frontend
    docker build -t backend ./backend
    
    ```
    
3. **Authenticate Docker to ECR**:
    
    ```bash
    aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
    
    ```
    
4. **Tag and Push Docker Images to ECR**:
    
    ```bash
    docker tag frontend:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/frontend-repo:latest
    docker tag backend:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/backend-repo:latest
    
    docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/frontend-repo:latest
    docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/backend-repo:latest
    
    ```
    

### AWS Deployment

1. **ECS Cluster and Service Configuration**:
    - **Cluster Name**: `cat-breed-detect-cluster`
    - **Frontend Service**:
        - **Image URI**: `<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/frontend-repo:latest`
        - **Container Port**: 80
        - **Protocol**: TCP
    - **Backend Service**:
        - **Image URI**: `<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/backend-repo:latest`
        - **Container Port**: 5000
        - **Protocol**: TCP
2. **Application Load Balancer (ALB)**:
    - **Listeners Configuration**:
        - **HTTP (80)**: Redirects to HTTPS (443).
        - **HTTPS (443)**: Forwards requests to target groups.
    - **Target Groups**:
        - **Frontend Target Group**:
            - **Protocol**: HTTP
            - **Port**: 80
            - **Health Check Path**: `/`
3. **Security Group Configuration**:
    - **Frontend Security Group**:
        - **Inbound Rules**:
            - Allow HTTP (80) and HTTPS (443) from `0.0.0.0/0`.
    - **Backend Security Group**:
        - **Inbound Rules**:
            - Allow HTTP (5000) from ALB's security group.
4. **Domain and HTTPS Setup**:
    - **Route 53**:
        - Create an A record pointing your custom domain to the ALB's DNS name.
    - **ACM**:
        - Request and validate an SSL/TLS certificate for your domain.
        - Attach the certificate to the ALB's HTTPS listener.

---

## Environment Variables

**Note**: In the AWS container, the environment variable `BACKEND_SERVICE_ENDPOINT` should be set to `http://localhost:5000`.

### Frontend

- `BACKEND_SERVICE_ENDPOINT`: URL of the backend service (e.g., `http://localhost:5000`).

### Backend

- No environment variables are currently required; configurations are static.

---

## Troubleshooting

1. **Health Check Failures**:
    - Ensure the target groups have the correct health check paths (Frontend: `/`).
    - Verify that security group rules allow traffic between the ALB and ECS containers.
2. **Unable to Access via HTTPS**:
    - Confirm that the ACM certificate is correctly attached to the ALB's HTTPS listener.
    - Ensure that the Route 53 A record correctly points to the ALB's DNS name.
3. **Environment Variable Issues**:
    - Double-check that the `BACKEND_SERVICE_ENDPOINT` in the frontend container is set to `http://localhost:5000` as per AWS container configuration.
    - In the Nginx configuration for the frontend, ensure that proxying is correctly set up to forward requests to the backend.

---

## Project Structure

```php
cat-breed-classifier/
├── frontend/                  # Frontend code and Dockerfile
│   ├── static/                # Static assets (images, CSS, etc.)
│   ├── index.html             # Main HTML page
│   ├── nginx.conf.template    # Nginx configuration template
│   └── Dockerfile             # Frontend Dockerfile
├── backend/                   # Backend code and Dockerfile
│   ├── app.py                 # Flask backend API
│   ├── cat_breed_model.h5     # Pre-trained model file
│   ├── cat_breed_class_indices.json  # Class labels
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Backend Dockerfile
├── docker-compose.yml         # Docker Compose file
└── README.md                  # Project documentation

```

---

## Usage Instructions

1. **Access the Application**:
    - Open your web browser and navigate to `https://catbreeddetect.com`
2. **Upload an Image**:
    - Use the web interface to upload a picture of a cat.
3. **View Prediction Results**:
    - Receive the predicted breed along with the confidence level.

---

## Result Display

Upload a picture of a cat to get the breed detection result.

![test1](/web_pic/web1.png)

![test2](/web_pic/web2.png)




