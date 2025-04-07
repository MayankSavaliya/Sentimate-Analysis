import requests
import json

url = "http://localhost:5000/predict"
data = {"text": "I am feeling really happy today"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
result = response.json()
print(result)