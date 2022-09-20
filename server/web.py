from flask import Flask , request ,Response
import logging
import datetime
import io
import os
from PIL import Image
from Model import Model
import torch
import torchvision.transforms as tr

call_cnt = 0

transform = tr.Compose(
    [   tr.Resize([256,256]),
        tr.ToTensor(),
     tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#from recycle import get_model
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SavedModel = 'ManyImageV5Rotation.pt'
device = 'cuda'

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

app = Flask(__name__)

def get_request_body_data(request):
    return request.get_data()

def Img_Save(image):
    image = io.BytesIO(image)
    image = image.open(image)
    image.save('save_img/save_img' + call_cnt + '.jpg','jpg')
    return image

def SavedModelLoad():
    model = Model()
    model = torch.load('ManyImageV5Rotation.pt')
    model.eval()
    return model

def ImgProcessing(image):
    image = transform(image)
    image = torch.unsqueeze(image,0)
    image = image.to(device)

@app.route('/',methods=['GET','POST'])
def handle_request( ):
    call_cnt += 1
    image = get_request_body_data(request)

    image = Img_Save(image)
    
    input = ImgProcessing(image)

    model = SavedModelLoad()

    outputs = model(input)

    logging.info(f"{outputs}")
    logging.info(f"{outputs.argmax(1).item()}")
    res = str(outputs.argmax(1).item())

    sss = 'default'

    if res == '0':
        sss = '내용물을 비우고 라벨을 제거하세요!'
    if res == '1':
        sss = '라벨을 제거하세요!'
    if res == '2':
        sss = '내용물을 비우세요!'
    if res == '3':
        sss = '쓰레기통에 버리시면 됩니다!'


    return Response(sss,mimetype='/text/plain; charset=utf-8')


if __name__ == '__main__':
    app.run('0.0.0.0',port=9090,debug=True)