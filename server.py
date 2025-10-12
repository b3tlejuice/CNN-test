from flask import Flask, render_template, request
import base64
from PIL import Image
import io
from corelib import Manipulator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save_image():
    image_data = request.json['image']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((96, 64), Image.NEAREST)
    image.save('out/image.png', 'PNG')
    
    return {'status': 'success'}

@app.route('/eval', methods=['POST'])
def eval_image():
    image_data = request.json['image']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((96, 64), Image.NEAREST).convert('L')
    dm = Manipulator()
    dm.uploadModel("out/model_weights.pth")
    result = dm.eval(image)
    return {'status': 'success', 'character' : result}


if __name__ == '__main__':
    app.run(debug=True)