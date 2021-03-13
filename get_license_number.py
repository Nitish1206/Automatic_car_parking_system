import cv2
import requests

def get_result_api(image):
    api_token = '8e5bd302366d69bdb9b2672008000405705f4294'
    url = 'https://api.platerecognizer.com/v1/plate-reader'
    headers = { 'Authorization': 'Token {0}'.format(api_token)}
    cv2.imwrite("temp.jpg", image)
    files = {'upload': open('temp.jpg', 'rb')}
    values = {'regions': 'gb'}
    data = requests.post(url, files=files, data=values, headers=headers)
    result = data.json()
    # print(15*"*")
    # print(data.json())
    # print(15*"*")
    if result.get("results"):
        plate = result['results'][0]['plate']
        return plate
    else:
        return None
        # types = result['results'][0]['vehicle']['type']
        # time = result.get('timestamp')
        # client.connect('127.0.0.1', 1883)
        # temp_data = {"status": 1, "msg": types+"  "+plate+"  "+datetime.now().strftime("%Y-%m-%d %H:%M")}
        # client.publish('parking/' + str(index), json.dumps(temp_data))
        # with open('result.txt', 'a+') as fp:
        #     fp.write("Parking-->"+str(index)+"-->"+types+"-->  "+plate+"   "+datetime.now().strftime("%Y-%m-%d %H:%M")+"\n")

