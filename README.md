# Retexo-flower-concept

This is a prototype implementation for training RetexoGNNs on edge devices. RetexoGNNs are the first neural networks for communication-efficient training on full-distributed graphs.
This prototype implementation is meant for training on two edge devices with a federated learning server. However, it can be easily extented to multiple edge devices. We have used flower framework to facilitate federated learning.
It has been developed with `Raspberry Pis` as the edge devices and `Linux x_86` machine as the server.

We have trained `RetexoGCN` on `Cora` dataset by dividing the dataset into two halves among the `Raspberry pis`. You can add your own model in `models.py` and similarly you can include your own dataset by adding appropriate file in `data/`. 
You can find the logic to split the dataset in `load_data.py` and change it according to your requirements too.

## Setting up Server
We suggest to use pip environment to manage the server. You can create a pip environment and install the requirements using the following commands

```
python -m venv env
source env/bin/activate 
pip install -r server-requirements.txt
```

To start the flower server for federated learning you can simply run the `run_server.sh` script. By default the server starts for `400` federated learning rounds on port `8080`. You can also start the server manually using the command
```
python server.py --num_rounds <NUM_ROUNDS> --port <PORT_NUMBER>
```
You can also make the following changes in the `run_server.sh`. These steps will get your flower server up and running. By the end of training it will display the `Train Loss` and `Validation Accuracy` for the training.

## Setting up the Client 
We highly recomment to use docker container to run clients on `Raspberry Pis`. We have used `RPi model 4B` with `Bluetooth 5.0 version`. You can use similar devices with bluetooth.

### Install docker on Raspberry Pi
You can look over the official Docker Documentation or follow the given commands to install docker on Raspberry Pis.

```
sudo apt-get update
sudp apt-get upgrade

sudo apt-get install apt-transport-https ca-certificates software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

Use this command to add docker repositories to your machine.
```
sudo add-apt-repository "deb [arch=<ARCH>] https://download.docker.com/linux/<OS_NAME> $(lsb_release -cs) stable"
```
You have to replace `<ARCH>` with the architecture of your edge devices like `arm64` and <OS_NAME> with the name of the operating system on your device.
An exmaple of the same command will be `sudo add-apt-repository "deb [arch=arm64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"`.

Follow these rest of the commands
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
```

Now you have successfully installed docker on your edge device.

## Building the Docker Image and Starting the Container
To build the image and start the container follow the given commands. Makesure to check the bluetooth service is inactive/off outside on your machine in order to access bluetooth from inside the container. You can check the same by this command 
```
systemctl status bluetooth
```
If the bluetooth service is active, stop the service using the command
```
systemctl stop bluetooth
```

Now, build the image and start the container
```
sudo docker build -t clientimage -f client-dockerfile .
sudo docker run --net=host --privileged -d -t --init clientimage
```
This will start the docker container in background. You can check the Container ID using `sudo docker ps`. 

## Bluetooth Pairing for both the edge devices
You have to pair the bluetooth of both devices for client-to-client communication over bluetooth. You can enter the container using the following command.
```
sudo docker exec -it container_name_or_id bash
```

Once you enter the container, start the dbus and bluetooth service inside the container
```
service dbus start
service bluetooth start
```

Now, you have to pair both the devices. You can refer [this](https://stackoverflow.com/questions/70424964/how-to-communicate-between-two-raspberry-pi-via-bluetooth-and-send-sensor-data-f) or follow the instruction given here only.

Run these commands on the first device
```
bluetoothctl
[bluetooth]# discoverable on
[bluetooth]# pairable on
[bluetooth]# agent on
[bluetooth]# default-agent
```

and these commands on the second device 
```
sudo bluetoothctl
[bluetooth]# discoverable on
[bluetooth]# pairable on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on
```
When the first device shows up in the scan of second device, pair them using this command with the mac address of first device in second `Raspberry Pi` 
```
[bluetoothctl]# pair XX:XX:XX:XX:XX:XX
```
You can check the device MAC address and name using `[bluetoothctl] show`. This is succesfully pair both the devices, now, you are ready to start training on your edge devices.

## Start training on edge devices
Given that your container is up and running. As mentioned earlier you can enter the container using `sudo docker exec -it container_name_or_id bash`.
To start the training you can use `run_client.sh`. You have to set the config in `run_client.sh` before starting the training. You need to `BLUETOOTH_SERVER_NAME` in the config to the bluetooth device of the `Raspberry Pi` acting as bluetooth server for client-to-client communication. You can randomly choose any one of your device for that set the server name observered earlier in the command `[bluetooth]# show`. Lastly you just need to add a flag `--bluetooth_server` in `python client.py` command of `run_client.sh` of the device you choose as bluetooth server.

| Argument  | Use     |
| :-------- |  :------------ |
|SERVER_ADDRESS  |  address of the flower server for training, defualt is set as `[::]:8080`|
|NUM_CLIENTS|  number of clients participating in the training|
|MODEL|  name of the model, train the models in this particular order only `mlp`, `mlppool_1` and `mlppool_2`|
|CID| client id|
|BLUETOOTH_SERVER_NAME| name of the bluetooth device acting as server (can be randomly among the two devices)|
|NUM_ROUNDS| number of federated learning round in training|

Lastly, as mentioned before, add the flag `--bluetooth_server` in `python client.py` command of the device acting as server.

Finally, you are ready to start training. Use the following command 
```
./run_clients.sh
```
After the training, change the `MODEL` in `run_clients.sh` to train all the models in the order of `mlp`, `mlppool_1`, `mlppool_2`. Meanwhile you have to use the same command in the server for all the models.


