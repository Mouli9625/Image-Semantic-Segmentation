from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)
app.secret_key = "1283371db0766d4482e60bb1b285a76d"

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_input(input_path):
    output_path = None
    caption = None  # Initialize caption variable
    # Load YOLO model
    model_yolo = YOLO("best.pt")  # segmentation model
    names = model_yolo.model.names

    # Check if input is a video or an image
    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # If input is a video
        cap = cv2.VideoCapture(input_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(os.path.join(app.config['UPLOAD_FOLDER'], 'instance-segmentation.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model_yolo.predict(im0)
            annotator = Annotator(im0, line_width=2)

            if results[0].masks is not None:
                clss = results[0].boxes.cls.cpu().tolist()
                masks = results[0].masks.xy
                for mask, cls in zip(masks, clss):
                    annotator.seg_bbox(mask=mask,
                                       mask_color=colors(int(cls), True),
                                       det_label=names[int(cls)])

            annotated_frame = annotator.result()
            out.write(annotated_frame)

        out.release()
        cap.release()
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'instance-segmentation.avi')
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # If input is an image
        im0 = cv2.imread(input_path)
        results = model_yolo.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])

        annotated_image = annotator.result()
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('segmented_' + os.path.basename(input_path)))
        cv2.imwrite(output_path, annotated_image)

        # Generate caption for the segmented image
        raw_image = Image.open(input_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        flash(f"Segmented image saved to: {output_path}")
        flash(f"Caption: {caption}")
    else:
        flash("Unsupported file format. Please provide a video (.mp4, .avi, .mov) or an image (.jpg, .jpeg, .png).")
    return output_path, caption

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    caption = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            segmented_path, caption = process_input(file_path)
            
            if segmented_path:
                return render_template('index.html', 
                                       uploaded_image=filename, 
                                       segmented_image=os.path.basename(segmented_path),
                                       caption=caption)

    return render_template('index.html', uploaded_image=uploaded_image, caption=caption)


@app.route('/show_output')
def show_output():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_image.png')
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
