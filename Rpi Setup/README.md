
# Raspberry Pi Setup Guide

### Hardware Requirements 
	- Raspberry Pi (Any) 
	- Power supply for Raspberry Pi 
	- MicroSD card (16GB or larger recommended) 
### Software Requirements 
	- Raspberry Pi OS (formerly Raspbian) 
  
## 1. Install Raspberry Pi OS
	- Download and install Raspberry Pi Imager from [raspberrypi.org](https://www.raspberrypi.org/software/) 
	- Use the imager to install Raspberry Pi OS on your MicroSD card 
	- Insert the MicroSD card into your Raspberry Pi and power it on
## 2. Install Required Packages
```bash
# Update package list
sudo  apt-get  update

# Install essential build tools
sudo  apt-get  install  build-essential  cmake  pkg-config
```
## 3. Configure Vim (Optional)

Create or modify **.vimrc** file for Vim customization. You can use a popular configuration like [amix/vimrc](https://github.com/amix/vimrc) or customize it following [this guide](https://www.freecodecamp.org/news/vimrc-configuration-guide-customize-your-vim-editor/).

## 4 . Set Static IP Address

Find your current network details:
```python
# Find current IP address
hostname -I

# Find default gateway (router) IP
ip r | grep default
```

Edit **dhcpcd.conf** to set a static IP address:
```python
sudo nano /etc/dhcpcd.conf
```
Add or modify the following lines (replace placeholders with your network details):

```python
interface wlan0 # Replace wlan0 with your network interface
static ip_address=STATIC_IP/24
static routers=ROUTER_IP
static domain_name_servers=DNS_IP
```

## 4. Configure WiFi

Edit **wpa_supplicant.conf** for WiFi configuration:

```python
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```
Add the following configuration (replace **YOUR_SSID** and **YOUR_PASS** with your WiFi credentials):

```python
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
ssid="YOUR_SSID"
scan_ssid=1
key_mgmt=WPA-PSK
psk="YOUR_PASS"
}
```

Save and exit the editor.

  

## 5. Restart and Verify

Ensure **rfkill** is disabled and reboot your Raspberry Pi:

```python
sudo rfkill unblock wifi
sudo reboot
```

After rebooting, your Raspberry Pi should connect to the configured WiFi network and use the static IP address.