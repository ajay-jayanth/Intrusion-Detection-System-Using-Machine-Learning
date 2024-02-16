import subprocess
import time

#get paths for later
from pathlib import Path
root_path = str(Path().absolute())
bin_path = root_path + "\\CICFlowmeter\\bin"
inputs_path = root_path + "\\inputs"
outputs_path = root_path + "\\outputs"
cfm_path = bin_path + "\\cfm.bat"

subprocess.call(["pktmon", "start", "--capture"])
time.sleep(10) #wait and read 10 seconds of internet traffic
subprocess.call(["pktmon", "stop"])
subprocess.call(["pktmon", "pcapng", "PktMon.etl", "-o", "inputs\\pktmon.pcap"])
subprocess.call(["rm", "PktMon.etl"])
subprocess.call([cfm_path, inputs_path, outputs_path], cwd=bin_path)
subprocess.call(["rm", "inputs\\pktmon.pcap"])

