
# Object Detection System Workflow

This project implements an object detection system using Faster R-CNN with a ResNet50 backbone. It supports both training and inference workflows, including an API for real-time prediction.

---

## **Project Structure**

### **Directories**
- `data/dataset.py`: Handles data loading and preprocessing.
- `model/detector.py`: Defines the object detection model architecture.
- `training/trainer.py`: Manages the training and validation processes.
- `inference/predictor.py`: Provides inference functionality for object detection.
- `train.py`: Main script for training the model using `trainer.py`.
- `app.py`: Implements an API for serving predictions.

---

## **System Workflow**

### **Training Phase**
1. **Load Dataset**:
   - Images and labels are loaded from the dataset.
   - Labels are converted from text format to tensors.
   - Necessary transformations (e.g., resizing, normalization) are applied.

2. **Initialize Model**:
   - Faster R-CNN architecture is initialized.
   - Backbone: ResNet50 for feature extraction.
   - Region Proposal Network (RPN) for object proposals.

3. **Train Model**:
   - Data is fed in batches using `DataLoader`.
   - Loss is calculated for classification and bounding box regression.
   - Optimizer updates model weights using backpropagation.

4. **Save Model**:
   - After training is complete, model weights are saved for future use.

---

### **Inference Phase**
1. **Load Trained Model**:
   - Saved model weights are loaded into the detector.

2. **Receive Input Image**:
   - The API accepts an image (JPG/PNG) via an endpoint.

3. **Preprocess Image**:
   - Resize and normalize the image to match the model's input requirements.

4. **Run Inference**:
   - The model predicts bounding boxes, class labels, and confidence scores.

5. **Format Predictions**:
   - Non-Maximum Suppression removes duplicate predictions.
   - Confidence thresholding filters out low-confidence results.
   - Output is returned as JSON.

---
## **Training Workflow**

### **`train.py`**

#### **Purpose**
- Serves as the main script for training the object detection model.

#### **Key Features**
1. **Configuration Loading**:
   - Reads training configurations from a YAML file (e.g., dataset paths, hyperparameters).
2. **Dataset Initialization**:
   - Prepares training and validation datasets using `AerialDataset`.
3. **Model Initialization**:
   - Loads the `AerialDetector` model with the specified number of classes.
4. **Training Process**:
   - Trains the model over multiple epochs using the `Trainer` class.
   - Logs training and validation losses for each epoch.
   - Saves model weights at each epoch.

#### **Usage**
```bash
python train.py -c <path_to_config.yaml>
```
---

## **Key Technical Aspects**
- **Architecture**: Faster R-CNN for object detection.
- **Feature Extraction**: ResNet50 backbone.
- **Region Proposal**: Region Proposal Network (RPN).
- **Feature Alignment**: ROI Pooling.
- **Prediction Refinement**:
  - Non-Maximum Suppression for duplicate removal.
  - Confidence thresholding for prediction filtering.

---

## **Data Format**

### **Input**:
- **Images**: JPG/PNG format.
- **Labels**: Text files with the following format:


### **Output**:
- **JSON**: Contains bounding boxes, class labels, and confidence scores:
```json
{
  "boxes": [[x1, y1, x2, y2], ...],
  "labels": [class_id, ...],
  "scores": [confidence_score, ...]
}
```

## **API Endpoints**

### **Implemented using FastAPI**

#### **Upload Image**
- Accepts an image file and returns predictions.
- Handles image preprocessing and model inference.

#### **Response Format**
- JSON response containing:
  - Bounding boxes
  - Labels
  - Confidence scores

---

## **Usage**

### **Training**
1. **Prepare the Dataset**:
   - Organize images and label files in the required format.

2. **Run the Training Script**:
   ```
   python train.py -c <path_to_config.yaml>

3. **Inference**:
    Start the API
   ```
   python app.py 
   ```
### **API Testing**

**Send an image to the API endpoint:**
Use tools like curl or Postman for testing.

**Test the API:**

Open your browser and navigate to http://0.0.0.0:8000/docs to access the Swagger UI.
Upload an image and execute the endpoint to receive predictions.