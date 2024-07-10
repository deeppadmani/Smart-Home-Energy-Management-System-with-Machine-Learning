import time
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import ssl
import json
import threading
import RPi.GPIO as GPIO
import pandas as pd

# GPIO setup
sensor_pin = 27
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(sensor_pin, GPIO.IN)  # Make sure the GPIO pin is set as input

# MQTT setup
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

client = mqtt.Client()
client.on_connect = on_connect
client.tls_set(ca_certs='./rootCA.pem',
               certfile='./e8a00db94abb24323eb3c8f3427c95b4bf9656d3d05507619bbe05747392b154-certificate.pem.crt',
               keyfile='./e8a00db94abb24323eb3c8f3427c95b4bf9656d3d05507619bbe05747392b154-private.pem.key',
               tls_version=ssl.PROTOCOL_TLS)
client.tls_insecure_set(True)
client.connect("a2cwtjb9gxadp9-ats.iot.us-east-1.amazonaws.com", 8883, 60)  # Use your own endpoint


def send_data_to_aws():
    entryno=0
    while True:
        current_time = datetime.now()
        if current_time - last_send_time >= timedelta(hours=1):
            # Read the power consumption data from the sensor
            power_data = read_ACS712()
            data_to_send = {
                'entryno': entryno,
                'Power_Consumption': power_data
            }
            message = json.dumps(data_to_send)
            client.publish("PowerConsumptionTable/entryno", payload=message, qos=0, retain=False)
            print(f"Sent data: {message}")

            # Update last send time
            last_send_time = current_time
            
            entryno = entryno + 1

def read_ACS712():
    # Read the ACS712 sensor value
    sensor_value = GPIO.input(sensor_pin)
    power_watts = sensor_value  # Adjust this line based on your actual sensor output
    power_kilowatts = power_watts / 1000.0

    # Assuming 1 hour interval (3600 seconds)
    energy_kWh = power_kilowatts * (3600 / 1000.0)
    print(f"Power Consumption (W): {power_watts}, Power Consumption (kW): {power_kilowatts:.2f} kW, Energy Consumption (kWh): {energy_kWh:.2f} kWh")
    return energy_kWh

if __name__ == "__main__":
    
    # Start the intrusion detector in a new thread
    thread = threading.Thread(target=send_data_to_aws)
    thread.start()

    # Keep the main thread alive
    client.loop_forever()