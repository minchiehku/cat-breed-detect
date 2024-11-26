{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8226d229-b1bd-4e9e-8d26-2e6f1e52369b",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (24.3.1)\n",
      "Requirement already satisfied: tensorflow in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (2.18.0)\n",
      "Requirement already satisfied: boto3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (1.35.54)\n",
      "Requirement already satisfied: numpy in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (1.26.4)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (24.3.25)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (0.6.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (0.2.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (18.1.1)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (3.4.0)\n",
      "Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (21.3)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (4.25.5)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (2.32.3)\n",
      "Requirement already satisfied: setuptools in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (75.1.0)\n",
      "Requirement already satisfied: six>=1.12.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (2.5.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (4.12.2)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (1.16.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (1.68.0)\n",
      "Requirement already satisfied: tensorboard<2.19,>=2.18 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (2.18.0)\n",
      "Requirement already satisfied: keras>=3.5.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (3.6.0)\n",
      "Requirement already satisfied: h5py>=3.11.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (3.12.1)\n",
      "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (0.4.1)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorflow) (0.37.1)\n",
      "Requirement already satisfied: botocore<1.36.0,>=1.35.54 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from boto3) (1.35.54)\n",
      "Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from boto3) (1.0.1)\n",
      "Requirement already satisfied: s3transfer<0.11.0,>=0.10.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from boto3) (0.10.3)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)\n",
      "Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from botocore<1.36.0,>=1.35.54->boto3) (2.9.0)\n",
      "Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from botocore<1.36.0,>=1.35.54->boto3) (2.2.3)\n",
      "Requirement already satisfied: rich in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (13.9.3)\n",
      "Requirement already satisfied: namex in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (0.13.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.10)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)\n",
      "Requirement already satisfied: markdown>=2.6.8 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.7)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.0.4)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from packaging->tensorflow) (3.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from rich->keras>=3.5.0->tensorflow) (2.18.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)\n"
     ]
    }
   ],
   "source": [
    "!pip install --upgrade pip\n",
    "!pip install tensorflow boto3 numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "09a51eb2-38d7-4ec7-99dc-d8b5432e71ce",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting kaggle\n",
      "  Downloading kaggle-1.6.17.tar.gz (82 kB)\n",
      "  Preparing metadata (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25hRequirement already satisfied: six>=1.10 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (1.16.0)\n",
      "Requirement already satisfied: certifi>=2023.7.22 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (2024.8.30)\n",
      "Requirement already satisfied: python-dateutil in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (2.9.0)\n",
      "Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (2.32.3)\n",
      "Requirement already satisfied: tqdm in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (4.66.5)\n",
      "Requirement already satisfied: python-slugify in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (8.0.4)\n",
      "Requirement already satisfied: urllib3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (2.2.3)\n",
      "Requirement already satisfied: bleach in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from kaggle) (6.1.0)\n",
      "Requirement already satisfied: webencodings in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from bleach->kaggle) (0.5.1)\n",
      "Requirement already satisfied: text-unidecode>=1.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from python-slugify->kaggle) (1.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from requests->kaggle) (3.4.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages (from requests->kaggle) (3.10)\n",
      "Building wheels for collected packages: kaggle\n",
      "  Building wheel for kaggle (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for kaggle: filename=kaggle-1.6.17-py3-none-any.whl size=105801 sha256=1dfc71f6945c4aad8fb13935129adaa60aad035bad68f3cfb986d65c79301f4a\n",
      "  Stored in directory: /home/ec2-user/.cache/pip/wheels/9f/af/22/bf406f913dc7506a485e60dce8143741abd0a92a19337d83a3\n",
      "Successfully built kaggle\n",
      "Installing collected packages: kaggle\n",
      "Successfully installed kaggle-1.6.17\n"
     ]
    }
   ],
   "source": [
    "!pip install kaggle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "99b4c6e2-d6bd-42f8-8137-b31921a3dfe4",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cp: cannot stat ‘/content/drive/MyDrive/kaggle.json’: No such file or directory\n",
      "chmod: cannot access ‘/home/ec2-user/.kaggle/kaggle.json’: No such file or directory\n",
      "Dataset URL: https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset\n",
      "License(s): DbCL-1.0\n",
      "Downloading cat-breeds-dataset.zip to /home/ec2-user/SageMaker\n",
      "100%|█████████████████████████████████████▉| 1.92G/1.93G [01:18<00:00, 22.8MB/s]\n",
      "100%|██████████████████████████████████████| 1.93G/1.93G [01:18<00:00, 26.4MB/s]\n"
     ]
    }
   ],
   "source": [
    "#上傳Kaggle的API憑證到Colab\n",
    "!mkdir ~/.kaggle\n",
    "!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/\n",
    "!chmod 600 ~/.kaggle/kaggle.json\n",
    "!kaggle datasets download -d ma7555/cat-breeds-dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0ee34527-40bd-4f85-888c-db7b23f46051",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current working directory: /home/ec2-user/SageMaker\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "print(\"Current working directory:\", os.getcwd())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "160c7537-668f-4304-849c-a61762d48ba2",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting script execution...\n",
      "Current working directory: /home/ec2-user/SageMaker\n",
      "Zip file found at './cat-breeds-dataset.zip'. Proceeding with extraction...\n",
      "Created directory: ./cat_breeds/\n",
      "Starting to extract files...\n",
      "Files successfully extracted to './cat_breeds/'\n",
      "Extracted files: ['images', 'data']\n"
     ]
    }
   ],
   "source": [
    "import zipfile\n",
    "import os\n",
    "\n",
    "print(\"Starting script execution...\")\n",
    "\n",
    "# 確認工作目錄\n",
    "print(\"Current working directory:\", os.getcwd())\n",
    "\n",
    "# 壓縮檔案的位置\n",
    "zip_file_path = './cat-breeds-dataset.zip'\n",
    "\n",
    "# 解壓縮目標目錄（暫存區）\n",
    "extract_to_path = './cat_breeds/'  # 解壓縮到當前目錄下\n",
    "\n",
    "# 確認壓縮檔是否存在\n",
    "if not os.path.exists(zip_file_path):\n",
    "    print(f\"Error: File '{zip_file_path}' not found. Please check the file path.\")\n",
    "else:\n",
    "    print(f\"Zip file found at '{zip_file_path}'. Proceeding with extraction...\")\n",
    "\n",
    "    try:\n",
    "        # 如果目標目錄不存在，則創建目錄\n",
    "        os.makedirs(extract_to_path, exist_ok=True)\n",
    "        print(f\"Created directory: {extract_to_path}\")\n",
    "\n",
    "        # 解壓縮\n",
    "        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:\n",
    "            print(\"Starting to extract files...\")\n",
    "            zip_ref.extractall(extract_to_path)\n",
    "            print(f\"Files successfully extracted to '{extract_to_path}'\")\n",
    "\n",
    "        # 確認解壓縮目錄內容\n",
    "        extracted_files = os.listdir(extract_to_path)\n",
    "        print(f\"Extracted files: {extracted_files}\")\n",
    "\n",
    "    except zipfile.BadZipFile as e:\n",
    "        print(f\"Error: Failed to unzip the file. {e}\")\n",
    "    except Exception as e:\n",
    "        print(f\"An unexpected error occurred: {e}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "678e79f1-d535-4b76-88c7-f53056f4a399",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 101314 images belonging to 67 classes.\n",
      "Found 25294 images belonging to 67 classes.\n",
      "Saved class indices to /tmp/cat_breed_class_indices.json\n",
      "Downloaded model checkpoint from S3: model/cat_breed_model_checkpoint.keras\n",
      "Loading model from checkpoint: /tmp/cat_breed_model_checkpoint.keras\n",
      "Found 101314 images belonging to 67 classes.\n",
      "Epoch 1/5\n",
      "\u001b[1m1781/3166\u001b[0m \u001b[32m━━━━━━━━━━━\u001b[0m\u001b[37m━━━━━━━━━\u001b[0m \u001b[1m10:43\u001b[0m 465ms/step - accuracy: 0.6734 - loss: 1.0459Skipping corrupted image batch due to error: broken data stream when reading image file\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 459ms/step - accuracy: 0.6685 - loss: 1.0643Found 25294 images belonging to 67 classes.\n",
      "\n",
      "Epoch 1: val_accuracy improved from -inf to 0.38074, saving model to /tmp/cat_breed_model_checkpoint.keras\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1779s\u001b[0m 560ms/step - accuracy: 0.6685 - loss: 1.0643 - val_accuracy: 0.3807 - val_loss: 5.1227\n",
      "Epoch 2/5\n",
      "\u001b[1m2369/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━\u001b[0m\u001b[37m━━━━━━\u001b[0m \u001b[1m5:56\u001b[0m 447ms/step - accuracy: 0.6756 - loss: 1.0397Skipping corrupted image batch due to error: broken data stream when reading image file\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 448ms/step - accuracy: 0.6730 - loss: 1.0494\n",
      "Epoch 2: val_accuracy did not improve from 0.38074\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1727s\u001b[0m 545ms/step - accuracy: 0.6730 - loss: 1.0494 - val_accuracy: 0.3782 - val_loss: 5.1547\n",
      "Epoch 3/5\n",
      "\u001b[1m 874/3166\u001b[0m \u001b[32m━━━━━\u001b[0m\u001b[37m━━━━━━━━━━━━━━━\u001b[0m \u001b[1m17:26\u001b[0m 456ms/step - accuracy: 0.6921 - loss: 0.9875Skipping corrupted image batch due to error: broken data stream when reading image file\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 449ms/step - accuracy: 0.6806 - loss: 1.0196\n",
      "Epoch 3: val_accuracy did not improve from 0.38074\n",
      "\u001b[1m3166/3166\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1736s\u001b[0m 548ms/step - accuracy: 0.6806 - loss: 1.0197 - val_accuracy: 0.3734 - val_loss: 5.2076\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as HDF5 format: /tmp/cat_breed_model.h5\n",
      "Uploaded /tmp/cat_breed_model.h5 to S3 as model/cat_breed_model.h5\n",
      "Uploaded /tmp/cat_breed_model_checkpoint.keras to S3 as model/cat_breed_model_checkpoint.keras\n",
      "Uploaded /tmp/cat_breed_class_indices.json to S3 as model/cat_breed_class_indices.json\n",
      "Training complete. Latest model checkpoint, HDF5 model, and class indices uploaded to S3.\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 1s/step\n",
      "Predicted breed: Exotic Shorthair with confidence: 0.44\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.layers import Dense, Flatten\n",
    "from tensorflow.keras.models import Model, load_model\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "import boto3\n",
    "import json\n",
    "import os\n",
    "import numpy as np\n",
    "from tensorflow.keras.preprocessing import image\n",
    "\n",
    "# 初始化 S3 客戶端\n",
    "s3 = boto3.client('s3')\n",
    "bucket_name = \"cat-breed-dataset-bucket\"\n",
    "\n",
    "# 本地檔案存儲路徑\n",
    "checkpoint_local_path = \"/tmp/cat_breed_model_checkpoint.keras\"\n",
    "class_indices_local_path = \"/tmp/cat_breed_class_indices.json\"\n",
    "\n",
    "# S3 檔案 Key\n",
    "checkpoint_s3_key = \"model/cat_breed_model_checkpoint.keras\"\n",
    "class_indices_s3_key = \"model/cat_breed_class_indices.json\"\n",
    "\n",
    "# 從 S3 獲取最新的檢查點文件\n",
    "def download_checkpoint_from_s3():\n",
    "    try:\n",
    "        s3.download_file(bucket_name, checkpoint_s3_key, checkpoint_local_path)\n",
    "        print(f\"Downloaded model checkpoint from S3: {checkpoint_s3_key}\")\n",
    "        return True\n",
    "    except Exception as e:\n",
    "        print(f\"No checkpoint found on S3 or error occurred: {e}\")\n",
    "        return False\n",
    "\n",
    "# 從 S3 下載類別索引文件\n",
    "def download_class_indices_from_s3():\n",
    "    try:\n",
    "        s3.download_file(bucket_name, class_indices_s3_key, class_indices_local_path)\n",
    "        print(f\"Downloaded class indices from S3: {class_indices_s3_key}\")\n",
    "    except Exception as e:\n",
    "        print(f\"Error downloading class indices from S3: {e}\")\n",
    "\n",
    "# 將最新的模型檢查點和類別索引上傳到 S3\n",
    "def upload_to_s3(local_path, s3_key):\n",
    "    try:\n",
    "        s3.upload_file(local_path, bucket_name, s3_key)\n",
    "        print(f\"Uploaded {local_path} to S3 as {s3_key}\")\n",
    "    except Exception as e:\n",
    "        print(f\"Error uploading {local_path} to S3: {e}\")\n",
    "\n",
    "# 定義一個安全的數據生成器，用於跳過損壞的圖片\n",
    "def safe_flow_from_directory(generator, directory, *args, **kwargs):\n",
    "    inner_generator = generator.flow_from_directory(directory, *args, **kwargs)\n",
    "    while True:\n",
    "        try:\n",
    "            batch_x, batch_y = next(inner_generator)\n",
    "            yield batch_x, batch_y\n",
    "        except Exception as e:\n",
    "            print(f\"Skipping corrupted image batch due to error: {e}\")\n",
    "\n",
    "# 設置數據生成器參數\n",
    "data_dir = './cat_breeds/images'  # 修改為解壓縮後的資料目錄\n",
    "img_size = (128, 128)  # 圖片大小\n",
    "batch_size = 32  # 批次大小\n",
    "\n",
    "# 使用 ImageDataGenerator 來進行數據增強和生成\n",
    "datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)\n",
    "\n",
    "# 創建原始的生成器\n",
    "original_train_generator = datagen.flow_from_directory(\n",
    "    data_dir,\n",
    "    target_size=img_size,\n",
    "    batch_size=batch_size,\n",
    "    class_mode='categorical',\n",
    "    subset='training'\n",
    ")\n",
    "\n",
    "original_validation_generator = datagen.flow_from_directory(\n",
    "    data_dir,\n",
    "    target_size=img_size,\n",
    "    batch_size=batch_size,\n",
    "    class_mode='categorical',\n",
    "    subset='validation'\n",
    ")\n",
    "\n",
    "# 使用安全生成器包裝原始生成器\n",
    "train_generator = safe_flow_from_directory(datagen, data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')\n",
    "validation_generator = safe_flow_from_directory(datagen, data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')\n",
    "\n",
    "# 獲取類別數量和樣本數量\n",
    "num_classes = len(original_train_generator.class_indices)\n",
    "steps_per_epoch = original_train_generator.samples // batch_size\n",
    "validation_steps = original_validation_generator.samples // batch_size\n",
    "\n",
    "# 保存類別索引到本地\n",
    "with open(class_indices_local_path, 'w') as f:\n",
    "    json.dump(original_train_generator.class_indices, f)\n",
    "print(f\"Saved class indices to {class_indices_local_path}\")\n",
    "\n",
    "# 從 S3 加載檢查點，如果存在\n",
    "if download_checkpoint_from_s3():\n",
    "    print(f\"Loading model from checkpoint: {checkpoint_local_path}\")\n",
    "    model = load_model(checkpoint_local_path)\n",
    "else:\n",
    "    print(\"No checkpoint found. Initializing a new model...\")\n",
    "    # 構建新模型\n",
    "    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))\n",
    "    x = Flatten()(base_model.output)\n",
    "    x = Dense(256, activation='relu')(x)\n",
    "    x = Dense(num_classes, activation='softmax')(x)  # 使用正確的類別數量\n",
    "    model = Model(inputs=base_model.input, outputs=x)\n",
    "\n",
    "    # 冷凍預訓練模型的層\n",
    "    for layer in base_model.layers:\n",
    "        layer.trainable = False\n",
    "\n",
    "# 編譯模型\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# 設置回調\n",
    "early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)\n",
    "checkpoint = ModelCheckpoint(\n",
    "    checkpoint_local_path,\n",
    "    monitor='val_accuracy',\n",
    "    save_best_only=True,\n",
    "    mode='max',\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "# 開始訓練模型\n",
    "history = model.fit(\n",
    "    train_generator,\n",
    "    validation_data=validation_generator,\n",
    "    epochs=5,\n",
    "    steps_per_epoch=steps_per_epoch,\n",
    "    validation_steps=validation_steps,\n",
    "    callbacks=[checkpoint, early_stopping]\n",
    ")\n",
    "\n",
    "# 定義 HDF5 模型的存檔路徑\n",
    "h5_model_path = \"/tmp/cat_breed_model.h5\"\n",
    "\n",
    "# 定義 HDF5 模型在 S3 上的目標檔案路徑\n",
    "h5_model_s3_key = \"model/cat_breed_model.h5\"\n",
    "\n",
    "# 保存模型為 HDF5 格式\n",
    "model.save(h5_model_path)\n",
    "print(f\"Model saved as HDF5 format: {h5_model_path}\")\n",
    "\n",
    "# 上傳最新的 HDF5 模型到 S3\n",
    "upload_to_s3(h5_model_path, h5_model_s3_key)\n",
    "\n",
    "# 訓練完成後，將最新檢查點和類別索引上傳回 S3\n",
    "upload_to_s3(checkpoint_local_path, checkpoint_s3_key)\n",
    "upload_to_s3(class_indices_local_path, class_indices_s3_key)\n",
    "\n",
    "print(\"Training complete. Latest model checkpoint, HDF5 model, and class indices uploaded to S3.\")\n",
    "\n",
    "\n",
    "# 測試模型的分類功能\n",
    "def classify_image(img_path, model, class_indices_path):\n",
    "    # 加載類別索引\n",
    "    with open(class_indices_path, 'r') as f:\n",
    "        class_labels = json.load(f)\n",
    "    class_labels = {v: k for k, v in class_labels.items()}  # 反轉字典以根據索引查找品種名稱\n",
    "\n",
    "    # 加載和預處理圖片\n",
    "    img = image.load_img(img_path, target_size=img_size)\n",
    "    img_array = image.img_to_array(img)\n",
    "    img_array = np.expand_dims(img_array, axis=0) / 255.0\n",
    "\n",
    "    # 進行預測\n",
    "    predictions = model.predict(img_array)\n",
    "    predicted_class = np.argmax(predictions, axis=1)[0]\n",
    "    breed_name = class_labels[predicted_class]\n",
    "    confidence = predictions[0][predicted_class]\n",
    "\n",
    "    print(f\"Predicted breed: {breed_name} with confidence: {confidence:.2f}\")\n",
    "\n",
    "# 使用測試圖片進行分類\n",
    "test_image_path = './test/test.jpg'  # 測試圖片路徑\n",
    "classify_image(test_image_path, model, class_indices_local_path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a48d27d4-55ae-4f62-bb3c-e26fd9686ef0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHFCAYAAAAaD0bAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABP4ElEQVR4nO3deVxU5eIG8OfMDAygQISyKAqoCKLmAipgamaCuFwtTdREFMxrpUnWNc0s9Vra4pKmeL2XJZcUDTG7Woo7KnrLgOznkpmGKURagqisc35/ECeGGZZhG4bzfD+f+Tjznve8531nOM3Te5YRRFEUQURERCQjCmN3gIiIiKixMQARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABEREZHsMAARERGR7DAAERERkewwABHJUFxcHARBgCAIOHbsmM5yURTRqVMnCIKAJ554ol63LQgCFi9ebPB6169fhyAIiIuLq/E658+fhyAIMDMzQ2ZmpsHbJKLmiwGISMasra0RHR2tU378+HFcvXoV1tbWRuhV/fnPf/4DACguLsbmzZuN3BsiakoYgIhkLCQkBAkJCcjNzdUqj46Ohr+/P9q3b2+kntVdQUEBtm3bhh49eqBt27aIiYkxdpcq9fDhQ/BnGYkaFwMQkYxNnDgRALB9+3apLCcnBwkJCQgPD9e7zu+//44XX3wRbdu2hbm5OTp06ICFCxeioKBAq15ubi6ef/552Nvbo2XLlhg2bBh++OEHvW1euXIFkyZNgoODA9RqNbp06YL169fXaWx79uzBnTt3MH36dISFheGHH37AyZMndeoVFBRg6dKl6NKlCywsLGBvb4/Bgwfj9OnTUh2NRoN169ahZ8+esLS0xCOPPAI/Pz/s3btXqlPZoT03NzdMnTpVel12+PHgwYMIDw9H69atYWVlhYKCAvz444+YNm0aPDw8YGVlhbZt22LUqFE4f/68Trt3797Fq6++ig4dOkCtVsPBwQHDhw/HpUuXIIoiPDw8EBQUpLNeXl4ebG1t8dJLLxn4jhI1LwxARDJmY2ODcePGac2ObN++HQqFAiEhITr18/PzMXjwYGzevBlz587Fvn37MHnyZLz//vt45plnpHqiKGLMmDHYsmULXn31VSQmJsLPzw/BwcE6bV64cAF9+vTB999/j5UrV+K///0vRowYgZdf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iA4ODqKVlZX4+OOPi8nJyeKgQYN0Poc//vhDnDNnjti+fXvRzMxMdHBwEEeMGCFeunRJp93FixeLAMQzZ85U+r4QyY0girz9KBFRc+br6wtBEPD1118buytETQYPgRERNUO5ubn4/vvv8d///hfnzp1DYmKisbtE1KQwABERNUPffvstBg8eDHt7e7z99tsYM2aMsbtE1KTwEBgRERHJDi+DJyIiItlhACIiIiLZYQAiIiIi2eFJ0HpoNBrcunUL1tbWem9XT0RERE2PKIq4d+9etb/xBzAA6XXr1i20a9fO2N0gIiKiWrhx40a1PxjMAKSHtbU1gNI30MbGxsi9ISIioprIzc1Fu3btpO/xqjAA6VF22MvGxoYBiIiIyMTU5PQVngRNREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywx9DJSIiogaj0Ygo0mhQVCKiqFgjPVcKApxsLYzWLwYgIiIiEyKKIoo1IopKNCgqLgsXpc8LS/58Lj1E6XlhsVjpsqISEYXFVSwr0ZSGFz3Liko0f223WESxRvNnW6XLijWi3nH0cbPDrpkBjfzu/YUBiIiIZK9E89cXeXHZl3qx7pd8cfkv/GLdAKC9vFz4qLisbDakwrLS1+UCRrEGRVLY+SuMmDpzlQJKhWDUPjAAERFRvSt/2KO4ki/96paVBopygaBYOyAUVwwL5WY5ijX6ZkT+muUo1pRur2x5JZMUJsNMKcBMqSj3ELT+NVdVLC+3rOy1SnfZX8sFmKkUMFNo1zMvV19V9lolVL5MKUCpECAIxg0/AAMQEZFJ0DrsoTVD0DiHPYo1ot4ZEUMPe5gKpULQHxAUumHBXM+XfGkd7WWldcuFDFXpa5Xir+flQ4y56s9lfz7XWqZUQFUu5DSFQGFqGICISLZKNHrOZTDwsIf2oQvtwx5FFZfJ+LCHIEB7NkH6kldApRC0AoH+mQzdEFBxNuOvWQ49y/5sv9Jl0vLSkKMw8uEZangMQETUoIrLzTQU/vnFXxYypLLyr7XKRBQWl0jhoaBCvYptlM1IlD4XddotqlDPxCcpKswC/Bke9IaAijMZCphJoaPi4YrSNrUOe5RbriofGP5cXpNlxj7fg6giBiCiZkCj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VSU6OhqTJ0/G5MmTpV+OL+///u//MGLECNjY2MDa2hoDBgzA1atXpeUxMTHo2rUr1Go1nJ2dMWvWLADA9evXIQgC0tLSpLp3796FIAg4duwYAODYsWMQBAEHDhyAr68v1Go1kpOTcfXqVYwePRqOjo5o2bIl+vTpo3NEqKCgAPPmzUO7du2gVqvh4eGB6OhoiKKITp064cMPP9Sq//3330OhUGj1vb4ZfA7Q6tWrta4CUygUaN26Nfr16wc7O7t67RwREdWBKAJFD4yzbTMroIorhsuoVCpMmTIFcXFxeOutt6Tvl127dqGwsBDPPfccHjx4AB8fH7z++uuwsbHBvn37EBoaig4dOqBfv37VbkOj0eCZZ55Bq1atcObMGeTm5uo9N8ja2hpxcXFo06YNzp8/j+effx7W1taYN28eQkJC8P333+Orr76SvtxtbW112njw4AGGDRsGPz8/fP3118jOzsb06dMxa9YsrZB39OhRODs74+jRo/jxxx8REhKCnj174vnnn690HFevXkVKSgp2794NURQRGRmJn376CR06dAAA3Lx5EwMHDsQTTzyBI0eOwMbGBqdOnZJmaaKiojB37lysWLECwcHByMnJwalTp6p9/yqaN28ePvzwQ3To0AGPPPIIfvnlFwwfPhzLli2DhYUFPvnkE4waNQqXL19G+/btAQBTpkxBSkoK1q5dix49euDatWu4ffs2BEFAeHg4YmNj8dprr0nbiImJwYABA9CxY0eD+1dTBgegqVOnNkA3iIio3hU9AN5tY5xtv3ELMG9RfT0A4eHh+OCDD3Ds2DEMHjwYQOkX4DPPPAM7OzvY2dlpfTnOnj0bX331FXbt2lWjAHTo0CFcvHgR169fl07VePfdd3XO23nzzTel525ubnj11VcRHx+PefPmwdLSEi1btoRKpYKTk1Ol29q2bRsePnyIzZs3Sz8V9fHHH2PUqFF477334OjoCKD0aMrHH38MpVIJLy8vjBgxAocPH64yAMXExCA4OFiabBg2bBhiYmKwbNkyAMD69etha2uLHTt2wMzMDADQuXNnaf1ly5bh1VdfxZw5c6SyPn36VPv+VbR06VIMHTpUem1vb48ePXpobScxMRF79+7FrFmz8MMPP2Dnzp1ISkqSzvcqC20AMG3aNLz11lv43//+h759+6KoqAhbt25t8KNKBh8Ci42N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mfj73/+OHTt2oG3bttBoNEhKSsK9e/fq3HEiIiKixlDjGSB9Ll++jOjoaGzZsgV3797F0KFDsXfv3vrsn1FwBoiIiMj0NMgMkD6enp54//338csvv2D79u11aYqIiIio0dRpBqi54gwQERGR6Wm0GSAiIiIiU8QARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESyY/QAtGHDBri7u8PCwgI+Pj5ITk6u0XqnTp2CSqVCz549tcrj4uIgCILOIz8/vwF6T0RERKbIqAEoPj4ekZGRWLhwIVJTUzFgwAAEBwcjIyOjyvVycnIwZcoUDBkyRO9yGxsbZGZmaj0sLCwaYghERERkgowagFatWoWIiAhMnz4dXbp0wZo1a9CuXTtERUVVud7f//53TJo0Cf7+/nqXC4IAJycnrQcRERFRGaMFoMLCQpw7dw6BgYFa5YGBgTh9+nSl68XGxuLq1at4++23K62Tl5cHV1dXuLi4YOTIkUhNTa23fhMREZHpUxlrw7dv30ZJSQkcHR21yh0dHZGVlaV3nStXrmD+/PlITk6GSqW/615eXoiLi0P37t2Rm5uLjz76CP3790d6ejo8PDz0rlNQUICCggLpdW5ubi1HRURERKbA6CdBC4Kg9VoURZ0yACgpKcGkSZOwZMkSdO7cudL2/Pz8MHnyZPTo0QMDBgzAzp070blzZ6xbt67SdZYvXw5bW1vp0a5du9oPiIiIiJo8owWgVq1aQalU6sz2ZGdn68wKAcC9e/fwzTffYNasWVCpVFCpVFi6dCnS09OhUqlw5MgRvdtRKBTo06cPrly5UmlfFixYgJycHOlx48aNug2OiIiImjSjHQIzNzeHj48PkpKS8PTTT0vlSUlJGD16tE59GxsbnD9/Xqtsw4YNOHLkCD777DO4u7vr3Y4oikhLS0P37t0r7YtarYZara7lSIiIiMjUGC0AAcDcuXMRGhoKX19f+Pv7Y9OmTcjIyMDMmTMBlM7M3Lx5E5s3b4ZCoUC3bt201ndwcICFhYVW+ZIlS+Dn5wcPDw/k5uZi7dq1SEtLw/r16xt1bERERNR0GTUAhYSE4M6dO1i6dCkyMzPRrVs37N+/H66urgCAzMzMau8JVNHdu3cxY8YMZGVlwdbWFr169cKJEyfQt2/fhhgCERERmSBBFEXR2J1oanJzc2Fra4ucnBzY2NgYuztERERUA4Z8fxv9KjAiIiKixsYARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREssMARERERLLDAERERESywwBEREREsmP0ALRhwwa4u7vDwsICPj4+SE5OrtF6p06dgkqlQs+ePXWWJSQkwNvbG2q1Gt7e3khMTKznXhMREZEpM2oAio+PR2RkJBYuXIjU1FQMGDAAwcHByMjIqHK9nJwcTJkyBUOGDNFZlpKSgpCQEISGhiI9PR2hoaEYP348zp4921DDICIiIhMjiKIoGmvj/fr1Q+/evREVFSWVdenSBWPGjMHy5csrXW/ChAnw8PCAUqnEnj17kJaWJi0LCQlBbm4uvvzyS6ls2LBhsLOzw/bt22vUr9zcXNja2iInJwc2NjaGD4yIiIganSHf30abASosLMS5c+cQGBioVR4YGIjTp09Xul5sbCyuXr2Kt99+W+/ylJQUnTaDgoKqbJOIiIjkRWWsDd++fRslJSVwdHTUKnd0dERWVpbe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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAioAAAHFCAYAAADcytJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA4e0lEQVR4nO3deXQUVd7G8aeyk5CEsCZI2GSJICASVkFkQBaRVwQHRESCKCObgw4iyqDgqOAo4jp49GVxBUSEcQaRRQEdEAUBhQEZxLD4AiI4JJBIgOS+f4Q06aQT0iGkbyffzzl90n3r1q3fTadOPamu7naMMUYAAAAWCvB1AQAAAAUhqAAAAGsRVAAAgLUIKgAAwFoEFQAAYC2CCgAAsBZBBQAAWIugAgAArEVQAQAA1iKoAOXEvHnz5DiOHMfR2rVr8y03xqhBgwZyHEc33HBDiW7bcRxNmTLF6/X27dsnx3E0b968IvV77rnnilcgAGsRVIByJjIyUrNnz87Xvm7dOu3du1eRkZE+qAoAPCOoAOXMwIEDtXjxYqWmprq1z549W+3bt1ft2rV9VBkA5EdQAcqZQYMGSZLmz5/vaktJSdHixYt19913e1zn119/1ahRo3TFFVcoJCRE9evX16RJk5SRkeHWLzU1Vffee6+qVKmiihUrqmfPnvrPf/7jccw9e/bojjvuUPXq1RUaGqqrrrpKr776agnN0rMDBw7ozjvvdNvmjBkzlJWV5dZv1qxZatGihSpWrKjIyEglJCTo0UcfdS1PT0/X+PHjVa9ePYWFhaly5cpKTEx0+50CKBlBvi4AQOmKiorSbbfdpjlz5ugPf/iDpOzQEhAQoIEDB+qFF15w63/69Gl16dJFe/fu1dSpU9W8eXN98cUXmjZtmrZt26Zly5ZJyr7GpW/fvtqwYYMee+wxtW7dWuvXr1evXr3y1bBz50516NBBtWvX1owZMxQbG6sVK1bo/vvv17Fjx/T444+X+Lx/+eUXdejQQWfOnNFf/vIX1a1bV//85z81fvx47d27V3/7298kSQsWLNCoUaM0duxYPffccwoICNAPP/ygnTt3usZ68MEH9fbbb+vJJ59Uy5YtlZaWph07duj48eMlXjdQ7hkA5cLcuXONJLNp0yazZs0aI8ns2LHDGGNM69atTVJSkjHGmKZNm5rOnTu71nvttdeMJPP++++7jffMM88YSWblypXGGGOWL19uJJkXX3zRrd9TTz1lJJnHH3/c1dajRw9Tq1Ytk5KS4tZ3zJgxJiwszPz666/GGGOSk5ONJDN37txC55bT79lnny2wz8SJE40k89VXX7m1jxw50jiOY3bv3u2qoVKlSoVu7+qrrzZ9+/YttA+AksFLP0A51LlzZ1155ZWaM2eOtm/frk2bNhX4ss9nn32miIgI3XbbbW7tSUlJkqRPP/1UkrRmzRpJ0uDBg9363XHHHW6PT58+rU8//VS33nqrwsPDde7cOdftpptu0unTp7Vx48aSmGa+eTRp0kRt2rTJNw9jjD777DNJUps2bXTixAkNGjRIf//733Xs2LF8Y7Vp00bLly/XxIkTtXbtWv32228lXi+AbAQVoBxyHEfDhg3TO++8o9dee02NGjVSp06dPPY9fvy4YmNj5TiOW3v16tUVFBTkernj+PHjCgoKUpUqVdz6xcbG5hvv3LlzevnllxUcHOx2u+mmmyTJYzi4VMePH1dcXFy+9po1a7qWS9KQIUM0Z84c7d+/X/3791f16tXVtm1brVq1yrXOSy+9pIcfflhLly5Vly5dVLlyZfXt21d79uwp8bqB8o6gApRTSUlJOnbsmF577TUNGzaswH5VqlTRzz//LGOMW/vRo0d17tw5Va1a1dXv3Llz+a7TOHLkiNvjmJgYBQYGKikpSZs2bfJ4ywksJalKlSo6fPhwvvZDhw5JkmsekjRs2DBt2LBBKSkpWrZsmYwxuvnmm7V//35JUkREhKZOnarvv/9eR44c0axZs7Rx40b16dOnxOsGyjuCClBOXXHFFXrooYfUp08fDR06tMB+Xbt21alTp7R06VK39rfeesu1XJK6dOkiSXr33Xfd+r333ntuj8PDw9WlSxdt3bpVzZs3V2JiYr5b3rMyJaFr167auXOntmzZkm8ejuO46s8tIiJCvXr10qRJk3TmzBn9+9//ztenRo0aSkpK0qBBg7R7926lp6eXeO1Aeca7foBybPr06Rftc9ddd+nVV1/V0KFDtW/fPjVr1kz/+te/9PTTT+umm25St27dJEndu3fX9ddfrwkTJigtLU2JiYlav3693n777Xxjvvjii+rYsaM6deqkkSNHqm7dujp58qR++OEH/eMf/3BdL+Kt7du364MPPsjX3rp1az3wwAN666231Lt3bz3x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1tVXX12s7RUVgSgb7/oBAPiFw4cPu+4vXLhQjz32mHbv3u1qy3mnSY6zZ88qODj4ouNWrlzZqzoCAwMVGxvr1TooPs6oAAD8QmxsrOsWHR0tx3Fcj0+fPq1KlSrp/fff1w033KCwsDC98847On78uAYNGqRatWopPDxczZo10/z5893GzfvST926dfX000/r7rvvVmRkpGrXrq3XX3/dtTzvmY61a9fKcRx9+umnSkxMVHh4uDp06OAWoiTpySefVPXq1RUZGal77rlHEydO1DXXXFPs30dGRobuv/9+Va9eXWFhYerYsaM2bdrkWv7f//5XgwcPVrVq1VShQgU1bNhQc+fOlZT9YX9jxoxRXFycwsLCVLduXU2bNq3YtVxOBBUAQPaHdJ0555ObMabE5vHwww/r/vvv165du9SjRw+dPn1arVq10j//+U/t2LFDI0aM0JAhQ/TVV18VOs6MGTOUmJiorVu3atSoURo5cqS+//77QteZNGmSZsyYoc2bNysoKEh33323a9m7776rp556Ss8884y++eYb1a5dW7NmzbqkuU6YMEGLFy/Wm2++qS1btqhBgwbq0aOHfv31V0nS5MmTtXPnTi1fvly7du3SrFmzVLVqVUnSSy+9pI8++kjvv/++du/erXfeeUd169a9pHoul2K99HPw4EE5jqNatWpJkr7++mu99957atKkiUaMGFGiBQIALr/fzmaqyWMrfLLtnU/0UHhIyVyJMG7cOPXr18+tbfz48a77Y8eO1SeffKJFixapbdu2BY5z0003adSoUZKyw8/MmTO1du1aJSQkFLjOU089pc6dO0uSJk6cqN69e+v06dMKCwvTyy+/rOHDh2vYsGGSpMcee0wrV67UqVOnijXPtLQ0zZo1S/PmzVOvXr0kSW+88YZWrVql2bNn66GHHtKBAwfUsmVLJSYmSpJbEDlw4IAaNmyojh07ynEc1alTp1h1lIZinVG54447tGbNGknSkSNHdOONN+rrr7/Wo48+qieeeKJECwQAoKhyDso5MjMz9dRTT6l58+aqUqWKKlasqJUrV+rAgQOFjtO8eXPX/ZyXmHI+Er4o6+R8bHzOOrt371abNm3c+ud97I29e/fq7Nmzuu6661xtwcHBatOmjXbt2iVJGjlypBYsWKBrrrlGEyZM0IYNG1x9k5KStG3bNjVu3Fj333+/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lxznot2CgoJ01113ad68eXrsscfknF9n0aJFOnPmjAYPHqz09HS1atVKDz/8sKKiorRs2TINGTJE9evXV9u2bS+6jaysLPXr109Vq1bVxo0blZqa6nY9S47IyEjNmzdPNWvW1Pbt23XvvfcqMjJSEyZM0MCBA7Vjxw598sknWr16tSQpOjo63xjp6enq2bOn2rVrp02bNuno0aO65557NGbMGLcwtmbNGsXFxWnNmjX64YcfNHDgQF1zzTW69957LzqfvIwx6tu3ryIiIrRu3TqdO3dOo0aN0sCBA7V27VpJ0uDBg9WyZUvNmjVLgYGB2rZtm4KDgyVJo0eP1pkzZ/T5558rIiJCO3fuVMWKFb2uwxvFCiqdO3fWf/7zH7366qv6/vvvZYxRv379NGLECE2ZMsX1PUDeyMzM1KJFi5SWlub6ADkAQCk5my49XdM32370kBQSUaSud999t5599lmtXbtWXbp0kZT9sk+/fv0UExOjmJgYjR8/3tV/7Nix+uSTT7Ro0aIiBZXVq1dr165d2rdvn2rVqiVJevrpp/NdV/LnP//Zdb9u3br605/+pIULF2rChAmqUKGCKlasqKCgIMXGxha4rXfffVe//fab3nrrLUVEZM//lVdeUZ8+ffTMM8+oRo0akqSYmBi98sorCgwMVEJCgnr37q1PP/20WEFl9erV+u6775ScnKz4+HhJ0ttvv62mTZtq06ZNat26tQ4cOKCHHnrIdYlHw4YNXesfOHBA/fv3V7NmzSRJ9evX97oGbxX7c1Rq1qypp556yq3t22+/1Ztvvqk5c+YUeZzt27erffv2On36tCpWrKglS5aoSZMmHvtmZGQoIyPD9Tg1NbV4xQMA/FJCQoI6dOigOXPmqEuXLtq7d6+++OILrVy5UlL2P73Tp0/XwoUL9X//93+u40ZOELiYXbt2qXbt2q6QIsnjP88ffPCBXnjhBf3www86deqUzp075/VXyOzatUstWrRwq+26665TVlaWdu/e7QoqTZs2VWBgoKtPXFyctm/f7tW2cm8zPj7eFVIkqUmTJqpUqZJ27dql1q1b68EHH9Q999yjt99+W926ddPvf/97XXnllZKk+++/XyNHjtTKlSvVrVs39e/f36tLPoqj2EGlpDRu3Fjbtm3TiRMntHjxYg0dOlTr1q3zGFamTZumqVOn+qBKACjjgsOz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kebPn6+wsDBdd9112rNnT4nWXxTFmWNcXJxef/11LV68WB9++KEaN26srl276vPPP3f1KSvP4eHDh7V8+XLdc889bu02PYfe8qd9sKTYvh96y5/2wUvlb/ugMUYPPvigOnbsqKuvvrrAfrbsh3797cnecBzH7bExJl/bxfrnbfd2zMupuLXMnz9fU6ZM0d///ndVr17d1d6uXTu1a9fO9fi6667Ttddeq5dfflkvvfRSyRXuBW/m2LhxYzVu3Nj1uH379jp48KCee+45XX/99cUa83Irbi3z5s1TpUqV1LdvX7d2G59Db/jbPngp/Gk/LCp/3AeLy9/2wTFjxui7777Tv/71r4v2tWE/LPNnVKpWrarAwMB86e7o0aP5UmCO2NhYj/2DgoJUpUqVQvsUNOblUpz55Vi4cKGGDx+u999/X926dSu0b0BAgFq3bu2T/wQuZY65tWvXzq3+svAcGmM0Z84cDRkyRCEhIYX29eVz6C1/2gcvlb/shyXB1n3wUvjbPjh27Fh99NFHWrNmjWrVqlVoX1v2wzIfVEJCQtSqVSutWrXKrX3VqlXq0KGDx3Xat2+fr//KlSuVmJio4ODgQvsUNOblUpz5Sdn/wSUlJem9995T7969L7odY4y2bdumuLi4S67ZW8WdY15bt251q9/fn0Mp+0r+H374QcOHD7/odnz5HHrLn/bBS+FP+2FJsHUfvBT+sg8aYzRmzBh9+OGH+uyzz1SvXr2LrmPNflhil+VabMGCBSY4ONjMnj3b7Ny504wbN85ERES4rs6eOHGiGTJkiKv/jz/+aMLDw80DDzxgdu7caWbPnm2Cg4PNBx984Oqzfv16ExgYaKZPn2527dplpk+fboKCgszGjRutn997771ngoKCzKuvvmoOHz7sup04ccLVZ8qUKeaTTz4xe/fuNVu3bjXDhg0zQUFB5quvvir1+Rnj/RxnzpxplixZYv7zn/+YHTt2mIkTJxpJZvHixa4+/vwc5rjzzjtN27ZtPY5p03N48uRJs3XrVrN161YjyTz//PNm69atZv/+/cYY/98HjfF+jv62H3o7P3/bB72dXw5/2QdHjhxpoqOjzdq1a93+3tLT0119bN0Py0VQMcaYV1991dSpU8eEhISYa6+91u0tWUOHDjWdO3d267927VrTsmVLExISYurWrWtmzZqVb8xFixaZxo0bm+DgYJOQkOC2A5Y2b+bXuXNnIynfbejQoa4+48aNM7Vr1zYhISGmWrVqpnv37mbDhg2lOKP8vJnjM888Y6688koTFhZmYmJiTMeOHc2yZcvyjemvz6Exxpw4ccJUqFDBvP766x7Hs+k5zHmrakF/c2VhH/R2jv62H3o7P3/bB4vzN+pP+6CnuUkyc+fOdfWxdT90zk8AAADAOmX+GhUAAOC/CCoAAMBaBBUAAGAtggoAALAWQQUAAFiLoAIAAKxFUAEAANYiqAAoUxzH0dKlS31dBoASQlABUGKSkpLkOE6+W8+ePX1dGgA/FeTrAgCULT179tTcuXPd2kJDQ31UDQB/xxkVACUqNDRUsbGxbreYmBhJ2S/LzJo1S7169VKFChVUr149LVq0yG397du363e/+50qVKigKlWqaMSIETp16pRbnzlz5qhp06YKDQ1VXFycxowZ47b82LFjuvXWWxUeHq6GDRvqo48+uryTBnDZEFQAlKrJkyerf//++vbbb3XnnXdq0KBB2rVrlyQpPT1dPXv2VExMjDZt2qRFixZp9erVbkFk1qxZGj16tEaMGKHt27fro48+UoMGDdy2MXXqVA0YMEDfffedbrrpJg0ePFi//vprqc4TQAkp0a84BFCuDR061AQGBpqIiAi32xNPPGGMyf4G1/vuu89tnbZt25qRI0caY4x5/fXXTUxMjDl16pRr+bJly0xAQIA5cuSIMcaYmjVrmkmTJhVYgyTz5z//2fX41KlTxnEcs3z58hKbJ4DSwzUqAEpUly5dNGvWLLe2ypUru+63b9/ebVn79u21bds2SdKuXbvUokULRUREuJZfd911ysrK0u7du+U4jg4dOqSuXbsWWkPz5s1d9yMiIhQZGamjR48Wd0oAfIigAqBERURE5Hsp5mIcx5EkGWNc9z31qVChQpHGCw4OzrduVlaWVzUBsAPXqAAoVRs3bsz3OCEhQZLUpEkTbdu2TWlpaa7l69evV0BAgBo1aqTIyEjVrVtXn376aanWDMB3OKMCoERlZGToyJEjbm1BQUGqWrWqJGnRokVKTExUx44d9e677+rrr7/W7NmzJUmDBw/W448/rqFDh2rKlCn65ZdfNHbsWA0ZMkQ1atSQJE2ZMkX33Xefqlevrl69eunkyZNav369xo4dW7oTBVAqCCoAStQnn3yiuLg4t7bGjRvr+++/l5T9jpwFCxZo1KhRio2N1bvvvqsmTZpIksLDw7VixQr98Y9/VOvWrRUeHq7+/fvr+eefd401dOhQnT59WjNnztT48eNVtWpV3XbbbaU3QQClyjHGGF8XAaB8cBxHS5YsUd++fX1dCgA/wTUqAADAWgQVAABgLa5RAVBqeKUZgLc4owIAAKxFUAEAANYiqAAAAGsRVAAAgLUIKgAAwFoEFQAAYC2CCgAAsBZBBQAAWIugAgAArPX/+tDSAZoGCSUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# 繪製準確率曲線\n",
    "plt.plot(history.history['accuracy'], label='Training Accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
    "plt.title('Model Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# 繪製損失曲線\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.title('Model Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "faf2a8ee-e195-40ba-8f8f-cd6352fb89ba",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_python3",
   "language": "python",
   "name": "conda_python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
