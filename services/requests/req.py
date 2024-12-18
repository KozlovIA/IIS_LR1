import requests
import time
import random

for i in range(50):
    params = {'mobile_id': i}
    data = {
        "battery_power":	random.randint(502, 1997),
        "blue":	     random.randint(0, 1),
        "clock_speed":	random.uniform(0.6, 2.9),
        "dual_sim":	random.randint(0, 1),
        "fc":	random.randint(1, 19),
        "four_g":	random.randint(0, 1),
        "int_memory":	random.randint(3, 63),
        "m_dep":	random.uniform(0.2, 0.9),
        "mobile_wt":	random.randint(81, 199),
        "n_cores":	random.randint(2, 7),
        "pc":	  random.randint(1, 19),
        "px_height": random.randint(1, 1959),
        "px_width": random.randint(501, 1997),
        "ram": random.randint(257, 3997),
        "sc_h": random.randint(6, 18),
        "sc_w": random.randint(1, 17),
        "talk_time": random.randint(3, 19),
        "three_g": random.randint(0, 1),
        "touch_screen": random.randint(0, 1),
        "wifi": random.randint(0, 1)
        } 
    response = requests.post('http://price-classifier:8000/api/prediction', params=params, json=data)
    time.sleep(random.randint(1,5))
    print(response.json())