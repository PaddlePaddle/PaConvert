import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from apibase import APIBase


def test(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    res = []
    for item in data:
        api = item["api"]
        obj = APIBase(api)
        torch_codes = item["test_code"]
        excepted_outputs = item["excepted_paddle_code"]
        for i in range(len(torch_codes)):
            code = torch_codes[i]
            excepted_output = excepted_outputs[i]
            obj.run(pytorch_code=code,expect_paddle_code=excepted_output,mode="min")
            


path = "tests/test_miss.json"
test(path)
