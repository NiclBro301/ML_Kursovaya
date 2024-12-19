import requests

url = 'http://127.0.0.1:5000/api/detect'
file = {'file': open('static/images/test_image.png', 'rb')}
response = requests.post(url, files=file)

print(response.json())