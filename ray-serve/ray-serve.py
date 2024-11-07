from ray import serve
from starlette.requests import Request
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from io import BytesIO
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

@serve.deployment
class ImageModel:
    def __init__(self):
        # Determine the device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Correctly set the path to your model checkpoint
        checkpoint_path = "/srv/nfs/kube-ray/storage/best-checkpoints/model.pt"  # Update this to your actual path

        # Initialize the model architecture
        self.model = resnet18()

        # Modify the fully connected layer to match the checkpoint dimensions
        num_classes = 2  # The number of classes in the saved checkpoint
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Load the saved weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        # Move the model to the selected device
        self.model = self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Load the labels.csv file
        self.labels_df = pd.read_csv("/srv/nfs/kube-ray/labels.csv")

        # Preprocessing steps
        self.preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    async def __call__(self, request: Request) -> Dict:
        try:
            # Read request JSON data
            request_data = await request.json()
            image_bytes = bytes.fromhex(request_data['image'])
            image_name = request_data.get('image_name')+".png"  # Do not add ".png" if not in CSV

            # Debug: Print the image name being looked up
            print(f"Image name to lookup: {image_name}")

            # Get the real label from labels.csv if the image_name is provided
            real_label = None
            matching_row = self.labels_df[self.labels_df['Part ID'] == image_name]
            if not matching_row.empty:
                real_label = matching_row.iloc[0]['Class']
                print(f"Found matching row: {matching_row}")

            # Preprocess the image
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.preprocessor(pil_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            # Get probabilities using softmax
            probabilities = F.softmax(output_tensor[0], dim=0)
            class_index = int(torch.argmax(probabilities))

            if class_index == 1:
                index="OK"
            else:
                index="NOK"
            # Prepare response
            response = {
                "predicted_class": index,
                "probabilities": probabilities.tolist()  # Convert tensor to list for JSON serialization
            }

            # If real label is found, check if it matches the predicted class
            if real_label is not None:
                response["real_label"] = real_label
                response["is_match"] = (index == real_label)

            return response
        except Exception as e:
            return {"error": str(e)}


image_model = ImageModel.bind()
app = image_model

serve.run(image_model)

