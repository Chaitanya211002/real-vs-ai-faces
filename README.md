# 🧑 vs 🤖 Real or AI-Generated Faces?

A game where you compete against a deep learning model to tell real human faces from AI-generated ones.

**Live demo:** [paste your Render URL here]
(Free hosting: the first load after inactivity can take about a minute.)

## The model
MobileNetV3Small pretrained on ImageNet, used as a frozen feature extractor with a small custom
classification head, trained on ~1,000 face images (128×128 RGB). Served as a TensorFlow Lite model
with the LiteRT runtime to fit the free hosting tier's memory limit.

## Model comparison (300-image held-out test set)
| Model | Test accuracy |
|---|---|
| MLP | 66.7% |
| CNN (from scratch) | 67.0% |
| CNN + data augmentation | 65.7% |
| **MobileNetV3Small (frozen)** | **76.0%** |
| MobileNetV3Small (fine-tuned) | 74.7% |

## How to play
1. Look at the face and click **Real** or **AI-generated**.
2. See the correct answer and the model's guess.
3. After all rounds, find out who won!

## Run locally
    pip install -r requirements.txt
    python app.py
Then open http://localhost:7860

## Files
- `app.py`: Gradio game
- `best_model.tflite`: model used by the app
- `best_model.keras`: original Keras model
- `test_images/`: 300 test images (128×128 PNG) + `labels.json`
