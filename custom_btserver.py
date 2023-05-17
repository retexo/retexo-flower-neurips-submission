from bluedot.btcomm import *
from time import sleep
from signal import pause
import pickle as pkl
import numpy as np
import sys
import struct
import time

class CustomBluetoothServer(BluetoothServer):
        def __init__(self,
                     data_received_callback,
                     max_bytes=50000,
                     auto_start=True,
                     device="hci0",
                     port=1,
                     encoding="utf-8",
                     power_up_device=False,
                     when_client_connects=None,
                     when_client_disconnects=None,):
                super().__init__(data_received_callback,auto_start,device,port,encoding,power_up_device,when_client_connects,when_client_disconnects)
                self.max_bytes = max_bytes
                self.start_time = time.time()

        def _read(self):
                new_message = True
                msglen = -1
                while self._client_connected:
                        if new_message:
                                raw_msglen = self.recvall(self._client_sock, 4)
                                new_message=False
                        if not raw_msglen:
                                print("Can't get the message length")
                                break
                        msglen = struct.unpack(">I", raw_msglen)[0]
                        print(f"Trying to read msg of len: {msglen}")
                        data = self.recvall(self._client_sock, msglen)
                        new_message=True
                        print(f"Received message of length: {msglen}, time spent so far {time.time() - self.start_time}")
                        if data:
                                if self._data_received_callback:
                                        if self._encoding:
                                                data = data.decode(self._encoding)
                                        self.data_received_callback(data)
                        if self._conn_thread.stopping.wait(BLUETOOTH_TIMEOUT):
                                break
                self._client_sock.close()
                self._client_sock = None
                self._client_info = None
                self._client_connected = False

        def recvall(self, sock, n):
                data = bytearray()
                while len(data) < n:
                        print(f"{len(data)} out of {n} bytes read...")
                        try:
                                packet = sock.recv(n - len(data))
                        except IOError as e:
                                self._handle_bt_error(e)
                                data = b""
                        data+=packet
                return data
