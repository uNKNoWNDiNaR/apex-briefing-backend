from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

from .align_signals import merge_topics


TOPIC_ALIASES = {
    '/constructor0/state_estimation': 'state',
    '/constructor0/can/wheels_speed_01': 'wheels',
    '/constructor0/can/psa_status_01': 'steer',
    '/constructor0/can/cba_status_fl': 'brake_fl',
    '/constructor0/can/cba_status_fr': 'brake_fr',
    '/constructor0/can/cba_status_rl': 'brake_rl',
    '/constructor0/can/cba_status_rr': 'brake_rr',
    '/constructor0/can/ice_status_02': 'ice',
    '/constructor0/can/kistler_acc_body': 'kistler_acc',
    '/constructor0/can/kistler_ang_vel_body': 'kistler_ang',
    '/constructor0/can/kistler_correvit': 'correvit',
    '/constructor0/can/mm710_tx1_z_ay': 'mm710_lat',
    '/constructor0/can/mm710_tx2_x_ax': 'mm710_lon',
    '/constructor0/can/mm710_tx3_y_az': 'mm710_vert',
    '/constructor0/can/badenia_560_tpms_front': 'tpms_front',
    '/constructor0/can/badenia_560_tpms_rear': 'tpms_rear',
    '/constructor0/can/badenia_560_tyre_surface_temp_front': 'surf_front',
    '/constructor0/can/badenia_560_tyre_surface_temp_rear': 'surf_rear',
    '/constructor0/can/badenia_560_brake_disk_temp': 'brake_temp',
    '/constructor0/can/badenia_560_wheel_load': 'wheel_load',
    '/constructor0/can/badenia_560_ride_front': 'ride_front',
    '/constructor0/can/badenia_560_ride_rear': 'ride_rear',
    '/constructor0/can/badenia_560_powertrain_press': 'power_press',
    '/constructor0/can/badenia_560_powertrain_temp': 'power_temp',
}

CAMERA_ALIASES = {
    '/constructor0/sensor/camera_fl/compressed_image': 'camera_fl',
    '/constructor0/sensor/camera_r/compressed_image': 'camera_r',
}


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if value is None:
        out[prefix[:-1]] = None
        return
    if isinstance(value, (str, bool, int, float)):
        out[prefix[:-1]] = value
        return
    if isinstance(value, np.generic):
        out[prefix[:-1]] = value.item()
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        out[f'{prefix[:-1]}_bytes'] = len(value)
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size <= 16:
            for idx, item in enumerate(value.tolist()):
                _flatten(item, f'{prefix}{idx}_', out)
        else:
            out[f'{prefix[:-1]}_len'] = int(value.size)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _flatten(item, f'{prefix}{idx}_', out)
        return
    if hasattr(value, '__dict__'):
        for key, item in vars(value).items():
            if key == '__msgtype__':
                continue
            if key == 'header' and hasattr(item, 'stamp'):
                out['header_stamp_sec'] = int(getattr(item.stamp, 'sec', 0))
                out['header_stamp_nanosec'] = int(getattr(item.stamp, 'nanosec', 0))
                out['header_frame_id'] = getattr(item, 'frame_id', '')
                continue
            _flatten(item, f'{prefix}{key}_', out)
        return
    out[prefix[:-1]] = value


def _message_timestamp_ns(message: Any, bag_timestamp_ns: int) -> int:
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is not None:
        sec = int(getattr(stamp, 'sec', 0))
        nanosec = int(getattr(stamp, 'nanosec', 0))
        if sec or nanosec:
            return sec * 1_000_000_000 + nanosec
    raw_ts = getattr(message, 'timestamp', None)
    if raw_ts is not None:
        try:
            return int(float(raw_ts) * 1_000_000_000)
        except Exception:
            pass
    return int(bag_timestamp_ns)


def _pick(df: pd.DataFrame, *columns: str, default=np.nan):
    for column in columns:
        if column in df.columns:
            return df[column]
    if hasattr(default, 'copy'):
        return default.copy()
    return default


def build_canonical_schema(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    out = df.copy()
    out['session_name'] = session_name
    out['timestamp'] = out['timestamp_ns'] / 1_000_000_000.0
    out['x'] = _pick(out, 'x_m')
    out['y'] = _pick(out, 'y_m')
    out['z'] = _pick(out, 'z_m')
    out['roll'] = _pick(out, 'roll_rad')
    out['pitch'] = _pick(out, 'pitch_rad')
    out['yaw'] = _pick(out, 'yaw_rad')
    out['heading'] = out['yaw']
    out['vx'] = _pick(out, 'vx_mps')
    out['vy'] = _pick(out, 'vy_mps')
    out['vz'] = _pick(out, 'vz_mps')
    out['speed'] = _pick(out, 'v_mps', 'v_raw_mps')
    out['yaw_rate'] = _pick(out, 'yaw_vel_rad', 'wz_radps')
    out['throttle'] = _pick(out, 'gas')
    out['brake'] = _pick(out, 'brake')
    out['steering'] = _pick(out, 'delta_wheel_rad', 'steer__psa_actual_pos_rad')
    out['gear'] = _pick(out, 'gear')
    out['rpm'] = _pick(out, 'rpm', 'ice__ice_engine_speed_rpm')
    out['brake_pressure_front'] = _pick(out, 'front_brake_pressure')
    out['brake_pressure_rear'] = _pick(out, 'rear_brake_pressure')
    out['brake_pressure_fl'] = _pick(out, 'cba_actual_pressure_fl_pa', 'brake_fl__cba_actual_pressure_fl_pa')
    out['brake_pressure_fr'] = _pick(out, 'cba_actual_pressure_fr_pa', 'brake_fr__cba_actual_pressure_fr_pa')
    out['brake_pressure_rl'] = _pick(out, 'cba_actual_pressure_rl_pa', 'brake_rl__cba_actual_pressure_rl_pa')
    out['brake_pressure_rr'] = _pick(out, 'cba_actual_pressure_rr_pa', 'brake_rr__cba_actual_pressure_rr_pa')
    out['wheel_speed_fl'] = _pick(out, 'wheels__wss_speed_fl_rad_s', 'omega_w_fl')
    out['wheel_speed_fr'] = _pick(out, 'wheels__wss_speed_fr_rad_s', 'omega_w_fr')
    out['wheel_speed_rl'] = _pick(out, 'wheels__wss_speed_rl_rad_s', 'omega_w_rl')
    out['wheel_speed_rr'] = _pick(out, 'wheels__wss_speed_rr_rad_s', 'omega_w_rr')
    out['slip_ratio_fl'] = _pick(out, 'lambda_fl_perc')
    out['slip_ratio_fr'] = _pick(out, 'lambda_fr_perc')
    out['slip_ratio_rl'] = _pick(out, 'lambda_rl_perc')
    out['slip_ratio_rr'] = _pick(out, 'lambda_rr_perc')
    out['slip_angle_fl'] = _pick(out, 'alpha_fl_rad')
    out['slip_angle_fr'] = _pick(out, 'alpha_fr_rad')
    out['slip_angle_rl'] = _pick(out, 'alpha_rl_rad')
    out['slip_angle_rr'] = _pick(out, 'alpha_rr_rad')
    out['ax'] = _pick(out, 'ax_mps2', 'mm710_lon__tx2_x_ax', 'kistler_acc__acc_x')
    out['ay'] = _pick(out, 'ay_mps2', 'mm710_lat__tx1_z_ay')
    out['az'] = _pick(out, 'az_mps2', 'mm710_vert__tx3_y_az')
    out['tyre_temp_fl'] = _pick(out, 'tpms_front__tpr4_temp_fl')
    out['tyre_temp_fr'] = _pick(out, 'tpms_front__tpr4_temp_fr')
    out['tyre_temp_rl'] = _pick(out, 'tpms_rear__tpr4_temp_rl')
    out['tyre_temp_rr'] = _pick(out, 'tpms_rear__tpr4_temp_rr')
    out['tyre_pressure_fl'] = _pick(out, 'tpms_front__tpr4_abs_press_fl')
    out['tyre_pressure_fr'] = _pick(out, 'tpms_front__tpr4_abs_press_fr')
    out['tyre_pressure_rl'] = _pick(out, 'tpms_rear__tpr4_abs_press_rl')
    out['tyre_pressure_rr'] = _pick(out, 'tpms_rear__tpr4_abs_press_rr')
    out['brake_temp_fl'] = _pick(out, 'brake_temp__brake_disk_temp_fl')
    out['brake_temp_fr'] = _pick(out, 'brake_temp__brake_disk_temp_fr')
    out['brake_temp_rl'] = _pick(out, 'brake_temp__brake_disk_temp_rl')
    out['brake_temp_rr'] = _pick(out, 'brake_temp__brake_disk_temp_rr')
    out['wheel_load_fl'] = _pick(out, 'wheel_load__load_wheel_fl')
    out['wheel_load_fr'] = _pick(out, 'wheel_load__load_wheel_fr')
    out['wheel_load_rl'] = _pick(out, 'wheel_load__load_wheel_rl')
    out['wheel_load_rr'] = _pick(out, 'wheel_load__load_wheel_rr')
    out['ride_height_front'] = _pick(out, 'ride_front__ride_height_front')
    out['ride_height_rear'] = _pick(out, 'ride_rear__ride_height_rear')
    out['damper_fl'] = _pick(out, 'ride_front__damper_stroke_fl')
    out['damper_fr'] = _pick(out, 'ride_front__damper_stroke_fr')
    out['damper_rl'] = _pick(out, 'ride_rear__damper_stroke_rl')
    out['damper_rr'] = _pick(out, 'ride_rear__damper_stroke_rr')
    out['engine_oil_temp'] = _pick(out, 'ice__ice_oil_temp_deg_c', 'power_temp__engine_oil_temperature')
    out['coolant_temp'] = _pick(out, 'ice__ice_water_temp_deg_c', 'power_temp__coolant_temperature')
    out['engine_oil_pressure'] = _pick(out, 'ice__ice_oil_press_k_pa', 'power_press__engine_oil_pressure')
    out['fuel_pressure'] = _pick(out, 'ice__ice_fuel_press_k_pa', 'power_press__fuel_press_direct')
    out['boost_pressure'] = _pick(out, 'power_press__boost_pressure')
    out['track_idx'] = _pick(out, 'sn_map_state_track_sn_state_sn_state_idx')
    out['track_ds'] = _pick(out, 'sn_map_state_track_sn_state_sn_state_ds')
    out['track_n'] = _pick(out, 'sn_map_state_track_sn_state_sn_state_n')
    out['track_inside'] = _pick(out, 'sn_map_state_track_sn_state_is_inside_borders')
    out['session_time_s'] = out['timestamp'] - float(out['timestamp'].iloc[0])
    return out


def ingest_session(mcap_path: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    mcap_path = Path(mcap_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topics_dir = output_dir / 'topics'
    topics_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = output_dir / 'camera'
    camera_dir.mkdir(parents=True, exist_ok=True)

    requested_topics = [config['ingest']['base_topic'], *config['ingest']['extra_topics']]
    topic_rows: dict[str, list[dict[str, Any]]] = {TOPIC_ALIASES[topic]: [] for topic in requested_topics}
    camera_rows: list[dict[str, Any]] = []
    available_topics: dict[str, str] = {}

    with AnyReader([mcap_path]) as reader:
        connections = [
            conn for conn in reader.connections
            if conn.topic in requested_topics or conn.topic in config['ingest']['camera_topics']
        ]
        for conn in connections:
            available_topics[conn.topic] = conn.msgtype
        for conn, bag_timestamp_ns, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, conn.msgtype)
            timestamp_ns = _message_timestamp_ns(message, bag_timestamp_ns)
            if conn.topic in CAMERA_ALIASES:
                camera_rows.append({
                    'timestamp_ns': timestamp_ns,
                    'bag_timestamp_ns': int(bag_timestamp_ns),
                    'camera_alias': CAMERA_ALIASES[conn.topic],
                    'camera_topic': conn.topic,
                    'format': getattr(message, 'format', ''),
                    'data_size_bytes': len(getattr(message, 'data', b'')),
                })
                continue
            alias = TOPIC_ALIASES[conn.topic]
            flat: dict[str, Any] = {
                'timestamp_ns': int(timestamp_ns),
                'bag_timestamp_ns': int(bag_timestamp_ns),
            }
            _flatten(message, '', flat)
            topic_rows[alias].append(flat)

    topic_frames: dict[str, pd.DataFrame] = {}
    for alias, rows in topic_rows.items():
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values('timestamp_ns').reset_index(drop=True)
            frame.to_parquet(topics_dir / f'{alias}.parquet', index=False)
        topic_frames[alias] = frame

    camera_index = pd.DataFrame(camera_rows).sort_values('timestamp_ns').reset_index(drop=True)
    if not camera_index.empty:
        camera_index.to_parquet(camera_dir / 'camera_index.parquet', index=False)

    base_alias = TOPIC_ALIASES[config['ingest']['base_topic']]
    base_df = topic_frames[base_alias]
    if base_df.empty:
        raise RuntimeError(f'No base telemetry found in {mcap_path}')
    extra_frames = {alias: frame for alias, frame in topic_frames.items() if alias != base_alias}
    merged = merge_topics(base_df, extra_frames, tolerance_ms=int(config['ingest']['merge_tolerance_ms']))
    session_name = mcap_path.stem.replace('hackathon_', '')
    canonical = build_canonical_schema(merged, session_name=session_name)
    telemetry_path = output_dir / 'session_telemetry.parquet'
    canonical.to_parquet(telemetry_path, index=False)

    metadata = {
        'session_name': session_name,
        'source_mcap': str(mcap_path),
        'rows': int(len(canonical)),
        'timestamp_start_ns': int(canonical['timestamp_ns'].iloc[0]),
        'timestamp_end_ns': int(canonical['timestamp_ns'].iloc[-1]),
        'available_topics': available_topics,
        'camera_rows': int(len(camera_index)),
    }
    (output_dir / 'session_metadata.json').write_text(json.dumps(metadata, indent=2))
    return {
        'session_name': session_name,
        'telemetry_path': str(telemetry_path),
        'camera_index_path': str(camera_dir / 'camera_index.parquet') if not camera_index.empty else None,
        'metadata_path': str(output_dir / 'session_metadata.json'),
    }
