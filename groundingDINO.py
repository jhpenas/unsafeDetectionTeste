from groundingdino.util.inference import  load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
import torch
import numpy as np
from PIL import Image

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed


CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'

model = load_model(CONFIG_PATH,WEIGHTS_PATH)

VIDEO_PATH = 'videos/boi.mp4'

TEXT_PROMPT = 'person with knife and without gloves'
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.15

OUTPUT_PATH = 'videos/output.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)

#Get Frame and Height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = "/videos/outputvideo1.mp4"

#Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #preprocess image
    transformed_image = preprocess_image(frame)

    #perform object detection on the frame
    boxes, logits, phrases = predict(
        model=model,
        image=transformed_image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

# Annotate the frame
    annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("GroundingDINO Tracking", annotated_frame)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
