import time
import paho.mqtt.client as mqtt
import ssl
import json
import threading
import RPi.GPIO as GPIO
import pandas as pd

# GPIO setup
#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
#GPIO.setup(21, GPIO.IN)  # Make sure the GPIO pin is set as input

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

# Data class
class PowerConsumptionData():
    def __init__(self, data) -> None:
        self.data = data

    def dataCleaning(self):
        self.data.drop(columns=["index", "Global_reactive_power", "Voltage", "Global_intensity",
                                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3","Date","Time"], inplace=True)


        self.data['Power_Consumption'] = self.data['Global_active_power'] * 60
        self.data.drop(columns=["Global_active_power"], inplace=True)
        # Add entryno column
        self.data['entryno'] = range(1, len(self.data) + 1)

def intrusionDetector(power_data):
        for index, row in power_data.iterrows():
            print("Intrusion Detected!")
            # Send the latest power consumption data to AWS
            latest_data = power_data.iloc[-1].to_dict()
            message = json.dumps(latest_data)
            client.publish("device/data", payload=message, qos=0, retain=False)
            time.sleep(5)

def send_data_to_aws():
    for index, row in data.iterrows():
        data_to_send = {
            'entryno': row['entryno'],
            'Power_Consumption': row['Power_Consumption']
        }
        message = json.dumps(data_to_send)
        client.publish("PowerConsumptionTable/entryno", payload=message, qos=0, retain=False)
        print(f"Sent data: {message}")
        time.sleep(5)  # Wait for 5 seconds before sending the next data point

if __name__ == "__main__":
    # Load and process data
    data = pd.read_csv('https://raw.githubusercontent.com/deeppadmani/Datasets/main/power_consumption/household_power_consumption.csv',
                low_memory=False, na_values=['nan','?'])
    data = data.ffill()

    PowerConsumptionDataObj = PowerConsumptionData(data=data)
    processed_data = PowerConsumptionDataObj.dataCleaning()

    # Start the intrusion detector in a new thread
    thread = threading.Thread(target=send_data_to_aws)
    thread.start()

    # Keep the main thread alive
    client.loop_forever()