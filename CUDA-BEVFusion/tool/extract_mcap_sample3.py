#!/usr/bin/env python3
"""
Extract a synchronized image and point cloud from an MCAP file.

This helper loads the first messages on the given topics *after* the
provided timestamp and saves them in the tensor format used by
BEVFusion.  The output directory will contain ``points.tensor``,
``images.tensor`` and a ``0-image.jpg`` file.

Timestamps must be provided in the form:
    YYYY-MM-DD H:MM:SS.sss AM/PM PDT
and will be converted to Unix seconds internally.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import cv2  # for Bayer demosaicing

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tensor import save

DTYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

def parse_pdt_timestamp(ts_str: str) -> float:
    """Parse a timestamp like '2025-04-07 4:52:16.773 PM PDT'"""
    # Strip trailing timezone label if present
    ts_str = ts_str.strip()
    # Expect literal 'PDT' at end
    if not ts_str.endswith('PDT'):
        raise ValueError(f"Timestamp must end with 'PDT': {ts_str}")
    # Parse datetime without timezone
    dt = datetime.strptime(ts_str, '%Y-%m-%d %I:%M:%S.%f %p PDT')
    # Attach Pacific Time (UTC-7)
    dt = dt.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
    # Convert to UTC and return seconds since epoch
    return dt.astimezone(ZoneInfo('UTC')).timestamp()

def pointcloud2_to_array(msg) -> np.ndarray:
    """Convert ``sensor_msgs/msg/PointCloud2`` to a structured array."""
    dtype = np.dtype({
        'names': [f.name for f in msg.fields],
        'formats': [DTYPE_MAP[f.datatype] for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='Path to ROS2 bag directory or MCAP file')
    parser.add_argument('--cam-topic', required=True, help='Image topic name')
    parser.add_argument('--lidar-topic', required=True, help='PointCloud2 topic name')
    parser.add_argument('--timestamp', type=str, required=True,
                        help='Timestamp in "YYYY-MM-DD H:MM:SS.sss AM/PM PDT" format')
    parser.add_argument('-o', '--out-dir', type=Path, default=Path('custom-example'),
                        help='Output directory')
    args = parser.parse_args()

    # Convert to nanoseconds
    ts_secs = parse_pdt_timestamp(args.timestamp)
    ts_ns = int(ts_secs * 1e9)

    # Open MCAP using rosbag2_py API
    reader = SequentialReader()
    storage_opts = StorageOptions(uri=str(args.bag), storage_id='mcap')
    conv_opts = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_opts, conv_opts)

    # Discover topic types
    topic_info = reader.get_all_topics_and_types()
    def find_type(topic_name: str) -> str:
        for info in topic_info:
            if info.name == topic_name:
                return info.type
        raise RuntimeError(f"Topic {topic_name} not in bag")

    cam_type = find_type(args.cam_topic)
    lidar_type = find_type(args.lidar_topic)

    cam_msg = None
    lidar_msg = None

    # Read until we get one message of each after timestamp
    while reader.has_next():
        topic, data, t = reader.read_next()
        if t < ts_ns:
            continue
        if topic == args.cam_topic and cam_msg is None:
            msg_cls = get_message(cam_type)
            cam_msg = deserialize_message(data, msg_cls)
        elif topic == args.lidar_topic and lidar_msg is None:
            msg_cls = get_message(lidar_type)
            lidar_msg = deserialize_message(data, msg_cls)
        if cam_msg is not None and lidar_msg is not None:
            break

    if cam_msg is None or lidar_msg is None:
        raise RuntimeError('No messages found after the requested timestamp')

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Handle image message
    if hasattr(cam_msg, 'format'):
        # CompressedImage
        img = Image.open(BytesIO(cam_msg.data.tobytes()))
    else:
        # Raw Image message
        enc = cam_msg.encoding.lower()
        raw = bytes(cam_msg.data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        h, w = cam_msg.height, cam_msg.width
        if enc == 'rgb8':
            arr = arr.reshape((h, w, 3)); mode = 'RGB'
        elif enc == 'bgr8':
            arr = arr.reshape((h, w, 3))[..., ::-1]; mode = 'RGB'
        elif enc == 'mono8':
            arr = arr.reshape((h, w)); mode = 'L'
        elif enc in ('mono16', '16uc1'):
            arr = arr.view(np.uint16).reshape((h, w)); mode = 'I;16'
        elif enc == 'bayer_rggb8':
            arr = arr.reshape((h, w))
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_BAYER_RG2BGR)
            arr = arr_bgr[..., ::-1]; mode = 'RGB'
        else:
            raise RuntimeError(f"Unsupported image encoding: {cam_msg.encoding}")
        img = Image.fromarray(arr, mode)
    img.save(args.out_dir / '0-image.jpg')

    # Preprocess image to BEVFusion resolution (256x704)
    img_tensor = np.asarray(img.resize((704, 256))).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)[None, None]
    save(img_tensor.astype(np.float16), args.out_dir / 'images.tensor')

    # Point cloud
    pc = pointcloud2_to_array(lidar_msg)
    fields = pc.dtype.names
    xyzi = np.stack([
        pc['x'], pc['y'], pc['z'],
        pc[fields[3]] if len(fields) > 3 else np.zeros(len(pc))
    ], axis=1)
    fifth = np.zeros((xyzi.shape[0], 1), dtype=np.float32)
    save(np.hstack([xyzi, fifth]).astype(np.float16), args.out_dir / 'points.tensor')


if __name__ == '__main__':
    main()
