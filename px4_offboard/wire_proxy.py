"""Run-local UDP relay recording actual MAVROS->PX4 control messages."""
import hashlib
import socket
import threading
import time


class WireProxy:
    def __init__(self,port=14582,px4_port=14581,mavros_port=14541):
        from pymavlink import mavutil
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1',port));self.socket.settimeout(.1)
        self.px4_port=px4_port;self.mavros_port=mavros_port;self.commands=[];self.error=None
        self.parsers={p:mavutil.mavlink.MAVLink(None) for p in (px4_port,mavros_port)}
        for parser in self.parsers.values():parser.robust_parsing=True
        self.stop=threading.Event();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:data,addr=self.socket.recvfrom(65535)
            except socket.timeout:continue
            except OSError:
                if not self.stop.is_set():self.error='UDP receive failed'
                return
            source=addr[1]
            if addr[0]!='127.0.0.1' or source not in self.parsers:continue
            destination=self.px4_port if source==self.mavros_port else self.mavros_port
            self.socket.sendto(data,('127.0.0.1',destination))
            if source!=self.mavros_port:continue
            for msg in self.parsers[source].parse_buffer(data) or []:
                if source==self.mavros_port and msg.get_type() in ('SET_ATTITUDE_TARGET','SET_POSITION_TARGET_LOCAL_NED'):
                    self.commands.append(dict(received_monotonic_s=time.monotonic(),message=msg.to_dict(),
                        mavlink_hex=msg.get_msgbuf().hex(),datagram_sha256=hashlib.sha256(data).hexdigest()))

    def close(self):
        self.stop.set();self.thread.join(timeout=2);self.socket.close()
