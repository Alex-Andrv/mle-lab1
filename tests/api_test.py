import requests
import glob
import time
import pytest

def wait_till_active():
    connected = False
    for _ in range(10):
        try:
            print("Testing: Trying to connect...")
            connected = requests.get('http://localhost:5000/').ok
            if connected: break
        except requests.exceptions.ConnectionError as e:
            time.sleep(1)
    return connected

@pytest.fixture(autouse=True)
def run_around_tests():
    assert wait_till_active()
    print("Connected")
    yield


def test_json_file():
    #print(requests.get('http://localhost:5000/get-test/5').json())
    headers = {'Content-Type': 'application/json'}
    for json_file in glob.glob("../tests/*.json"):
        res = requests.post('http://localhost:5000/predict', data=open(json_file, 'rb'), headers=headers)
        assert res.ok
        print(res.json())