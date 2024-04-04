import rclpy
from rclpy.node import Node

import serial
import numpy as np

from tmf882x_interfaces.msg import TMF882XMeasure

class TMF882XPub(Node):
    def __init__(self):
        super().__init__('tmf882x_pub')

        self.TMF882X_CHANNELS = 10
        self.TMF882X_BINS = 128
        self.TMF882X_SKIP_FIELDS = 3 # skip first 3 items in each row
        self.TMF882X_IDX_FIELD = 2 # second item in each row contains the idx field

        ARDUINO_PORT = "/dev/ttyACM0"
        BAUDRATE = 1000000

        self.arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUDRATE, timeout=0.1)

        self.get_logger().info(f"Arduino port: {self.arduino.name}, baudrate: {self.arduino.baudrate}")

        self.publisher = self.create_publisher(TMF882XMeasure, 'tmf882x', 1)

        self.timer = self.create_timer(0.005, self.timer_callback)

        self.received_data = False

    def timer_callback(self):
        m = self.get_measurement()
        if m is None:
            return
        hists, dists, histogram_type = m
        if not self.received_data:
            self.get_logger().info("Received data from Arduino")
            self.received_data = True

        message = TMF882XMeasure()
        message.num_zones = self.TMF882X_CHANNELS-1 # -1 because ignore reference hist
        message.num_bins = self.TMF882X_BINS
        message.i2c_address = dists[0]["I2C_address"]
        message.tick = dists[0]["tick"]
        message.num_valid_results = dists[0]["num_valid_results"]
        message.temperature = dists[0]["temperature"]
        message.measurement_num = dists[0]["measurement_num"]
        message.depths_1 = dists[0]["depths_1"]
        message.confs_1 = dists[0]["confs_1"]
        message.depths_2 = dists[0]["depths_2"]
        message.confs_2 = dists[0]["confs_2"]
        message.histogram_type = histogram_type
        if hists[0]: # if hists is not an empty list (histograms are being reported)
            message.hists = np.array(hists[0][1:]).flatten().tolist()
            message.reference_hist = hists[0][0]
        
        self.publisher.publish(message)

        

    def process_raw_hists(self, buffer):
        if len(buffer) != 31:
            self.get_logger().warn("Buffer wrong size ({}) - measurement skipped".format(len(buffer)))
            return None, None

        rawSum = [[0 for _ in range(self.TMF882X_BINS)] for _ in range(self.TMF882X_CHANNELS)]

        for i, line in enumerate(buffer):
            data = line.decode('utf-8')
            data = data.replace('\r','')
            data = data.replace('\n','')
            row = data.split(',')

            if len(row) > 0 and len(row[0]) > 0 and row[0][0] == "#": # if it's a well formed line

                if row[0] == '#Raw' and len(row) == self.TMF882X_BINS+self.TMF882X_SKIP_FIELDS: # if it is a full "raw" histogram
                    if '' in row:
                        print("Empty entry recieved over serial - skipping and returning None")
                        return None
                    histogram_type = "full"
                    idx = int(row[self.TMF882X_IDX_FIELD]) # idx is the id of the histogram (e.g. 0-9 for 9 hists + calibration hist)
                    if ( idx >= 0 and idx <= 9 ):
                        for hist_bin in range(self.TMF882X_BINS):
                            rawSum[idx][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin])                          # LSB is only assignement
                    elif ( idx >= 10 and idx <= 19 ):
                        for hist_bin in range(self.TMF882X_BINS):
                            rawSum[idx - 10][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin]) * 256               # mid
                    elif ( idx >= 20 and idx <= 29 ):
                        for hist_bin in range(self.TMF882X_BINS):
                            rawSum[idx - 20][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin]) * 256 * 256         # MSB
                    else:
                        self.get_logger().error("Line read with invalid idx")
                
                elif row[0] == '#Prt': # if it is a partial histogram
                    histogram_type = "partial"
                    idx = int(row[self.TMF882X_IDX_FIELD]) # idx is the id of the histogram (e.g. 0-9 for 9 hists + calibration hist)
                    # when a partial histogram is printed, the value after the idx field is the
                    # number of bins which were skipped
                    skipped_bins = int(row[self.TMF882X_IDX_FIELD+1])
                    if ( idx >= 0 and idx <= 9 ):
                        for hist_bin in range(skipped_bins, len(row)-self.TMF882X_SKIP_FIELDS):
                            rawSum[idx][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin])
                    elif ( idx >= 10 and idx <= 19 ):
                        for hist_bin in range(skipped_bins, len(row)-self.TMF882X_SKIP_FIELDS):
                            rawSum[idx - 10][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin]) * 256
                    elif ( idx >= 20 and idx <= 29 ):
                        for hist_bin in range(skipped_bins, len(row)-self.TMF882X_SKIP_FIELDS):
                            rawSum[idx - 20][hist_bin] += int(row[self.TMF882X_SKIP_FIELDS+hist_bin]) * 256 * 256
                    else:
                        self.get_logger().error("Line read with invalid idx")

            else:
                self.get_logger().warn("Incomplete line read - measurement skipped")

        return rawSum, histogram_type

    def process_raw_dist(self, buffer):

        for line in buffer:
            data = line.decode('utf-8')
            data = data.replace('\r','')
            data = data.replace('\n','')
            d = data.split(',')

            # if there is an #Obj tag but no info in it, then the #Obj tag is just being used
            # as a separator between histograms, so return an empty distance result
            if d[0] == "#Obj" and len(d) == 1:
                return {
                    "I2C_address": 0,
                    "measurement_num": 0,
                    "temperature": 0,
                    "num_valid_results": 0,
                    "tick": 0,
                    "depths_1": [0 for _ in range(9)],
                    "confs_1": [0 for _ in range(9)],
                    "depths_2": [0 for _ in range(9)],
                    "confs_2": [0 for _ in range(9)]
                }

            if d[0] == "#Obj" and len(d) == 78:
                result = {}
                result["I2C_address"] = int(d[1])
                result["measurement_num"] = int(d[2])
                result["temperature"] = int(d[3])
                result["num_valid_results"] = int(d[4])
                result["tick"] = int(d[5])
                result["depths_1"] = [int(x) for x in [
                    d[6], d[8], d[10],
                    d[12], d[14], d[16],
                    d[18], d[20], d[22]
                ]]
                result["confs_1"] = [int(x) for x in [
                    d[7], d[9], d[11],
                    d[13], d[15], d[17],
                    d[19], d[21], d[23]
                ]]
                #18 that go in between here are unused, at least in 3x3 mode
                result["depths_2"] = [int(x) for x in [
                    d[42], d[44], d[46],
                    d[48], d[50], d[52],
                    d[54], d[56], d[58]
                ]]
                result["confs_2"] = [int(x) for x in [
                    d[43], d[45], d[47],
                    d[49], d[51], d[53],
                    d[55], d[57], d[59]
                ]]
                # last 18 are unused, at least in 3x3 mode

                return result
        return None

    def get_measurement(self):
        buffer = []
        frames_finished = 0 # start at -1 to throw out the first frame

        all_processed_hists = []
        all_processed_dists = []

        while frames_finished < 1:
            line = self.arduino.readline().rstrip()
            if line != "":
                buffer.append(line)
            try:
                decoded_line = line.decode('utf-8').rstrip().split(',')
                if decoded_line[0] == "#Obj":
                    if len(buffer) > 1: # if histograms were reported between #Obj (depth) measurements
                        processed_hists, histogram_type = self.process_raw_hists(buffer)
                    else:
                        processed_hists = []
                        histogram_type = "None"
                    processed_dists = self.process_raw_dist(buffer)
                    if processed_hists is not None and processed_dists is not None:
                        if frames_finished > -1:
                            all_processed_hists.append(processed_hists)
                            all_processed_dists.append(processed_dists)
                        frames_finished += 1
                    buffer = []

            except UnicodeDecodeError:
                self.get_logger().warn("UnicodeDecodeError - measurement skipped")
                pass # if you start in a weird place you get random data that can't be decoded, so just ignore
                buffer = []

        return all_processed_hists, all_processed_dists, histogram_type
        

def main(args=None):
    rclpy.init(args=args)
    tmf882x_pub = TMF882XPub()
    rclpy.spin(tmf882x_pub)

    tmf882x_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
