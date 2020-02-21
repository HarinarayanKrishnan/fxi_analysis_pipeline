import os
import zmq
import numpy as np
import collections
import time
import argparse
import intake
from collections import defaultdict
from event_model import DocumentRouter
from bluesky.callbacks import LiveTable as LiveTable_
from databroker._drivers.msgpack import BlueskyMsgpackCatalog

import asyncio
import zmq.asyncio

import copy
import uuid

import event_model


class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentError(
                self,
                "msgpack directory:{0} is not a valid path".format(prospective_dir),
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentError(
                self,
                "msgpack directory:{0} is not a readable dir".format(prospective_dir),
            )


"""
Some Test data
"""
# python document_to_zmq_replay.py -o 5010 msgpack -m /home/hari/data/hari/Tomography/NSLS-II-FXI-18/export-4XXXX/ -p Andor
# data_dir = "/home/hari/data/hari/Tomography/NSLS-II-FXI-18/export-1XXXX"
# data_dir = "/home/hari/data/hari/Tomography/NSLS-II-FXI-18/export-4XXXX"
# data_dir = "/home/hari/data/hari/Tomography/NSLS-II-FXI-18/export-9XXX"


class LiveTable(LiveTable_, DocumentRouter):
    """
    Patch up LiveTable to make it aware of the paginated types, event_page and
    datum_page. The upcoming release of bluesky will fix this.
    """

    pass


class FxiStandardFlyScan(DocumentRouter):
    """
    Base FxiFlyScan parser. Converts Documents into a specific Schema expected by Subscriber
    Pipeline. See fxi_analysis Dockerfile.
    """

    def __init__(self, catalog, run, zmq_socket, root_map):
        self.catalog = catalog
        self.run = run
        self.socket = zmq_socket
        self.root_map = root_map

        self.stream_count = defaultdict(lambda: 0)

        self.scan_id = None
        self.x_eng = None
        self.chunk_size = None

        self.dark_avg = None
        self.data = []
        self.white_avg = None
        self.white_ts = None
        self.descriptor_uids = {}
        self.image_angle = None

        self._buffered_thetas = []
        self._theta_timestamps = []
        self._image_timestamps = []

        self.counter = 0
        self._file = None
        self._dataset1 = None
        self._dataset2 = None
        self.data_to_timestamp_map = {}

        self._fpp = 20

    def read_timestamps(self, filename):
        import h5py

        _key = [
            "/entry/instrument/NDAttributes/NDArrayEpicsTSSec",
            "/entry/instrument/NDAttributes/NDArrayEpicsTSnSec",
        ]
        self._file = filename
        self._dataset1 = None
        self._dataset2 = None

        # Don't read out the dataset until it is requested for the first time.
        with h5py.File(filename, "r") as _file:
            self._dataset1 = np.array(_file[_key[0]])
            self._dataset2 = np.array(_file[_key[1]])

    def return_timestamp(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        start, stop = point_number * self._fpp, (point_number + 1) * self._fpp
        rtn = self._dataset1[start:stop].squeeze()
        rtn = rtn + (self._dataset2[start:stop].squeeze() * 1e-9)
        return rtn

    def reset_data(self):
        """
        Reset the data to enable more than one use
        """
        self.counter = 0
        self._file = None
        self._dataset1 = None
        self._dataset2 = None
        self.data_to_timestamp_map = {}

        self.stream_count = defaultdict(lambda: 0)

        self.scan_id = None
        self.x_eng = None
        self.chunk_size = None

        self.dark_avg = None
        self.data = []
        self.white_avg = None
        self.white_ts = None
        self.descriptor_uids = {}
        self.image_angle = None

        self._buffered_thetas = []
        self._theta_timestamps = []
        self._image_timestamps = []

    def start(self, doc):
        self.scan_id = doc["scan_id"]

        try:
            self.x_eng = doc["XEng"]
        except:
            self.x_eng = doc["x_ray_energy"]

        self.chunk_size = doc["chunk_size"]

        parameters = {
            "parameters": {
                "scan_id": self.scan_id,
                "x_eng": self.x_eng,
                "chunk_size": self.chunk_size,
            }
        }

        print("sending params", parameters, doc)
        self.socket.send_pyobj(parameters)

    def datum_page(self, doc):
        for i, datum_id in enumerate(doc["datum_id"]):
            point_number = doc["datum_kwargs"]["point_number"][i]
            print(
                datum_id, point_number, doc["resource"], doc.keys(), doc["datum_kwargs"]
            )
            self.data_to_timestamp_map[datum_id] = point_number

    def descriptor(self, doc):
        if doc["name"] == "baseline":
            self.descriptor_uids["baseline"] = doc["uid"]
        elif doc["name"] == "primary":
            self.descriptor_uids["primary"] = doc["uid"]
            file_name = doc["configuration"]["Andor"]["data"][
                "Andor_hdf5_full_file_name"
            ]

            rml = list(self.root_map.items())
            file_name = file_name.replace(rml[0][0], rml[0][1])
            self.read_timestamps(file_name)

            # print(self._dataset1)
            # print(self._dataset2)
            # print(file_name)

        elif doc["name"] == "zps_pi_r_monitor":
            self.descriptor_uids["zps_pi_r_monitor"] = doc["uid"]

    def event(self, doc):
        self.stream_count[doc["descriptor"]] += 1

        if doc["descriptor"] == self.descriptor_uids.get("primary"):

            """
            if there is another descriptor then it means that saved last frame
            from last iteration needs to pushed now.
            """

            # print(doc.keys(), doc["uid"], doc["time"], doc["timestamps"]["Andor_image"], doc["descriptor"], doc["data"].keys(), doc['filled'])
            ts_array = self.return_timestamp(
                self.data_to_timestamp_map[doc["filled"]["Andor_image"]]
            )
            print(ts_array)

            if self.white_avg is not None:
                # print("Last Data:", self.white_avg.shape)
                self.socket.send_pyobj(
                    {"tomo_index": self.counter, "tomo": self.white_avg}
                )
                self._image_timestamps.append(self.white_ts)
                self.white_avg = None
                self.white_ts = None
                self.counter = self.counter + 1

            if self.stream_count[doc["descriptor"]] == 1:
                dark_avg = np.mean(doc["data"]["Andor_image"][0], axis=0, keepdims=True)
                # print("Dark:", dark_avg.shape)
                self.socket.send_pyobj({"dark_avg": dark_avg})
                start_from = 1
            else:
                start_from = 0

            # print(doc["data"].keys(), doc["descriptor"], self.descriptor_uids.values())
            print(len(doc["data"]["Andor_image"][start_from:-1]))
            for image in doc["data"]["Andor_image"][start_from:-1]:
                # print("Data:", image.shape)
                self.socket.send_pyobj({"tomo_index": self.counter, "tomo": image})
                self.counter = self.counter + 1

            for ts in ts_array[start_from:-1]:
                self._image_timestamps.append(ts)

            self.white_avg = doc["data"]["Andor_image"][-1]
            self.white_ts = ts_array[-1]
        elif doc["descriptor"] == self.descriptor_uids.get("zps_pi_r_monitor"):
            # print(doc["data"], self._buffered_thetas)
            self._buffered_thetas.append(doc["data"]["zps_pi_r"])
            self._theta_timestamps.append(doc["timestamps"]["zps_pi_r"])

    def stop(self, doc):
        if self.white_avg is not None:
            self.white_avg = np.mean(self.white_avg, axis=0, keepdims=True)
            print("Ending:", self.white_avg.shape)
            self.socket.send_pyobj({"white_avg": self.white_avg})

            EPICS_TO_UNIX_OFFSET = 631152000  # 20 years in seconds

            thetas = np.interp(
                [item + EPICS_TO_UNIX_OFFSET for item in self._image_timestamps],
                self._theta_timestamps,
                self._buffered_thetas,
            )

            self.socket.send_pyobj({"thetas": thetas})
            print("SENDING", thetas)


class FxiDocumentStreamToZMQ:
    """
    Convert Fxi Documents to form that can be shipped over ZMQ
    to MPIArray based analysis routine
    """

    def __init__(self, publish_port):
        """
        Initialize Bluesky msgpack catalog based on prefix...
        """

        self.catalog = None
        self.run = None

        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.PUSH)

        print("Starting binding...", publish_port)
        if int(publish_port) == 0:
            publish_port = self.socket.bind_to_random_port("tcp://*")
        else:
            self.socket.bind("tcp://*:" + str(publish_port))

        self.publish_port = publish_port

        print("Listening on port: ", self.publish_port)
        # event_loop = asyncio.get_running_loop()
        # print("STARTING")
        # asyncio.ensure_future(self.write_to_zmq_remote_pub(publish_uri), loop=event_loop)
        # asyncio.ensure_future(self.write_to_zmq_remote_pub(publish_uri))

    """
    async def write_to_zmq_remote_pub(self, remote_uri):
        socket = self.ctx.socket(zmq.PUB)
        print("Starting binding...")
        socket.bind(remote_uri)
        print("Ending binding...")
    """

    async def read_from_zmq_triggered(self, uri):
        """
        ZMQ SUB wait for Stop Document
        """
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://" + uri)

    async def read_from_msgpack(self, data_dir, data_prefix):
        """
        Read information from Msgpack on disk using BluskyMsgpackCatalog
        """

        # self.reset_data()

        root_map = {"/NSLS2/xf18id1/DATA/Andor": f"{data_dir}/{data_prefix}"}

        print("Mapping", data_dir, root_map)

        self.catalog = BlueskyMsgpackCatalog(f"{data_dir}/*.msgpack", root_map=root_map)
        # print(list(self.catalog))
        self.run = self.catalog[-1]

        await self.replay(self.run.canonical(fill="yes"), root_map)

    async def read_from_mongodb(self, mongodb):
        pass

    async def replay(self, gen, root_map):
        """
        Emit documents to a callback with realistic time spacing.

        Parameters
        ----------
        gen: iterable
            Expected to yield (name, doc) pairs
        callback: callable
            Expected signature: callback(name, doc)
        """

        DOCUMENTS_WITHOUT_A_TIME = {"datum", "datum_page", "resource"}
        cache = collections.deque()
        name, doc = next(gen)

        if name != "start":
            raise ValueError("Expected gen to start with a RunStart document")

        """
        Set up the appropriate callback emitter
        """
        if doc["plan_name"] == "user_fly_only":
            print("TODO: create different reader")

        callback = FxiStandardFlyScan(self.catalog, self.run, self.socket, root_map)

        # Compute time difference between now and the time that this run started.
        offset = time.time() - doc["time"]
        callback(name, doc)
        for name, doc in gen:
            if name in DOCUMENTS_WITHOUT_A_TIME:
                # The bluesky RunEngine emits these documents immediately
                # before emitting an Event, which does have a time. Lacking
                # more specific timing info, we'll cache these and then emit
                # them in a burst before the next document with an associated time.
                cache.append((name, doc))
            else:
                delay = max(0, offset - (time.time() - doc["time"]))
                time.sleep(delay)
                while cache:
                    # Emit any cached documents without a time in a burst.
                    callback(*cache.popleft())
                # Emit this document.
                callback(name, doc)


global fxi_zmq


async def main():
    global fxi_zmq
    parser = argparse.ArgumentParser(
        description="FXI-18 Document Stream to ZMQ Stream Transformer",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-o", "--output_zmq_publish_port", type=int, default=0, required=True
    )
    sp = parser.add_subparsers()
    sp_msgpack = sp.add_parser("msgpack", help="Msgpack option")
    sp_msgpack.add_argument(
        "-m", "--msgpack_directory", action=readable_dir, required=True
    )
    sp_msgpack.add_argument("-p", "--prefix", required=True)

    sp_socket = sp.add_parser("zmq", help="ZMQ Socket option")
    sp_socket.add_argument("-u", "--uri", required=True)

    results = vars(parser.parse_args())

    print("Arguments:", results)
    fxi_zmq = FxiDocumentStreamToZMQ(results["output_zmq_publish_port"])

    if "uri" in results:
        await fxi_zmq.read_from_zmq_triggered(results["uri"])

    if "msgpack_directory" in results:
        await fxi_zmq.read_from_msgpack(results["msgpack_directory"], results["prefix"])


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
