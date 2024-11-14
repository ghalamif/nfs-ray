import gradio as gr
import requests
from PIL import Image
from io import BytesIO

# Function to interact with the Ray Serve deployment
def classify_image(input_image, f_name):
    try:
        # Convert image to bytes
        buffered = BytesIO()
        input_image.save(buffered, format="png")
        image_bytes = buffered.getvalue()
        image_hex = image_bytes.hex()

        # Set a placeholder image name (since input_image doesn't have a filename when uploaded)
        image_name = f_name  # Placeholder if no filename is available

        # Prepare JSON payload
        payload = {
            "image": image_hex,
            "image_name": image_name
        }

        print(f"Image name: {image_name}")
        
        # Send request to Ray Serve deployment
        response = requests.post("http://127.0.0.1:8000/ImageModel", json=payload)

        # Process response
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                return f"Error: {result['error']}", None  # Ensure two outputs are returned even if there's an error

            output_text = f"Predicted Class Index: {result['predicted_class']}\n"
            output_text += f"Probabilities: {result['probabilities'][0]} and {result['probabilities'][1]}\n"

            if "real_label" in result:
                output_text += f"Real Label: {result['real_label']}\n"
                if result['is_match'] == True:
                    color = "green"
                else:
                    color = "red"
                output_text += f"**Prediction Match:** <span style='color:{color};'>{result['is_match']}</span>\n"
            # Construct probabilities dictionary
            probabilities_dict = {"NOK": result['probabilities'][0], "OK": result['probabilities'][1]}
            
            return output_text, probabilities_dict
        else:
            return f"Request failed with status code {response.status_code}", None
    except Exception as e:
        return f"Error: {str(e)}", None  # Ensure two outputs are returned even if there's an exception

# Launch Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=[gr.Image(type="pil"), "text"],
    outputs=["text", gr.Label(num_top_classes=2)],
    title="Image Classification with ResNet18",
    examples=[
        ["/srv/nfs/kube-ray/visionline/1094.png"],
        ["/srv/nfs/kube-ray/visionline/1219.png"],
        ["/srv/nfs/kube-ray/visionline/1220.png"],
        ["/srv/nfs/kube-ray/visionline/1221.png"],
        ["/srv/nfs/kube-ray/visionline/1222.png"],
        ["/srv/nfs/kube-ray/visionline/1223.png"],
        ["/srv/nfs/kube-ray/visionline/1224.png"],
    ]
)

# Start Gradio independently
#interface.launch(server_name="0.0.0.0", share=False)
interface.launch(server_name="0.0.0.0", share=False, allowed_paths=["/srv/nfs/kube-ray/visionline"])
