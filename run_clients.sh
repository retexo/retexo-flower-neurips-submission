#!/bin/bash

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=2
MODEL="mlppool_2"
CID=1
BLUETOOTH_SERVER_NAME="ubuntupi2-desktop"
NUM_ROUNDS=400

echo "Starting client(cid=$CID) with partition out of $NUM_CLIENTS clients."

python client.py --model=$MODEL --cid=$CID --server_address=$SERVER_ADDRESS --nb_clients=$NUM_CLIENTS --bluetooth_server_name=$BLUETOOTH_SERVER_NAME --num_rounds=$NUM_ROUNDS