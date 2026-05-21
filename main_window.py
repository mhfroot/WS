# ui/main_window.py
import sys
import os
import time
import subprocess
import json
import re
from pathlib import Path
import config as app_config
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia
from PyQt5.QtCore import Qt

try:
    import smbus
except Exception:
    smbus = None

from sdr import BladeRFPowerMeter
from gsm_channels import (
    ChannelMode,
    channel_mode_measurement_note,
    channel_mode_uses_shared_tdd_carrier,
    channel_to_freq_mhz,
    lte_b40_earfcn_range,
)
from config import (
    DBM_MIN,
    DBM_MAX,
    DEFAULT_BANDWIDTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_GAIN,
    POWER_SMOOTH_ALPHA,
    TDD_UPLINK_ASSIST_ENABLED,
    TDD_UPLINK_ASSIST_SMOOTH_ALPHA,
)

SOURCE_UI_DIR = Path(__file__).resolve().parent
SOURCE_APP_DIR = SOURCE_UI_DIR.parent
RUNTIME_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_APP_DIR))
IMG_DIR = RUNTIME_DIR / "img"
WAV_DIR = RUNTIME_DIR / "wav"


def _cfg_number(name: str, default, cast):
    try:
        return cast(getattr(app_config, name, default))
    except Exception:
        return cast(default)


INA219_I2C_BUS_ID = _cfg_number(
    "INA219_I2C_BUS_ID",
    getattr(app_config, "I2C_BUS_ID", 1),
    int,
)
INA219_I2C_ADDR = _cfg_number("INA219_I2C_ADDR", 0x40, int)
INA219_SHUNT_OHMS = _cfg_number("INA219_SHUNT_OHMS", 0.1, float)
INA219_CURRENT_IDLE_THRESHOLD_MA = _cfg_number(
    "INA219_CURRENT_IDLE_THRESHOLD_MA",
    30.0,
    float,
)
BATTERY_VOLTAGE_EMPTY_MV = _cfg_number("BATTERY_VOLTAGE_EMPTY_MV", 9000, int)
BATTERY_VOLTAGE_FULL_MV = _cfg_number("BATTERY_VOLTAGE_FULL_MV", 12600, int)


class PowerMeterWorker(QtCore.QObject):
    """
    Worker di thread terpisah untuk baca power dari SDR.
    """
    powerUpdated = QtCore.pyqtSignal(float)
    spectrumUpdated = QtCore.pyqtSignal(object)
    tddMetricsUpdated = QtCore.pyqtSignal(object)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, sdr: BladeRFPowerMeter, parent=None):
        super().__init__(parent)
        self.sdr = sdr
        self._running = False
        self._last_spectrum_ts = 0.0

        # Spectrum config (updated by UI thread).
        self._cfg_lock = QtCore.QMutex()
        self.spectrum_fft_size = 2048
        self.spectrum_averages = 3
        self.spectrum_interval_s = 0.25
        self.tdd_uplink_assist = False

    @QtCore.pyqtSlot()
    def start(self):
        self._running = True
        while self._running:
            try:
                self._cfg_lock.lock()
                try:
                    fft_size = int(self.spectrum_fft_size)
                    averages = int(self.spectrum_averages)
                    interval_s = float(self.spectrum_interval_s)
                    use_tdd_uplink_assist = bool(self.tdd_uplink_assist)
                finally:
                    self._cfg_lock.unlock()

                now = time.monotonic()
                need_spectrum = (now - self._last_spectrum_ts) >= interval_s
                tdd_metrics = None

                if need_spectrum:
                    if use_tdd_uplink_assist:
                        tdd_metrics = self.sdr.measure_tdd_uplink_metrics()
                        p_dbm = float(tdd_metrics.get("display_dbm", DBM_MIN))
                        spec_dbm = self.sdr.measure_spectrum_dbm(
                            fft_size=fft_size,
                            averages=averages,
                        )
                    else:
                        p_dbm, spec_dbm = self.sdr.measure_power_and_spectrum_dbm(
                            fft_size=fft_size,
                            averages=averages,
                        )
                else:
                    if use_tdd_uplink_assist:
                        tdd_metrics = self.sdr.measure_tdd_uplink_metrics()
                        p_dbm = float(tdd_metrics.get("display_dbm", DBM_MIN))
                    else:
                        p_dbm = self.sdr.measure_power_dbm()
                    spec_dbm = None
                self.powerUpdated.emit(p_dbm)
                if use_tdd_uplink_assist and tdd_metrics is not None:
                    self.tddMetricsUpdated.emit(tdd_metrics)

                if need_spectrum:
                    self._last_spectrum_ts = now
                    self.spectrumUpdated.emit(spec_dbm)
            except Exception as e:
                self.errorOccurred.emit(str(e))
                break

    @QtCore.pyqtSlot()
    def stop(self):
        self._running = False


class WaterfallWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dbm = np.nan
        self._hold_dbm = np.nan
        self._max_hold_enabled = False
        self._history_dbm = []
        self._history_bias = []
        self._history_capacity = 420
        self._segment_points = 6
        self._pending_point_bias = 0.0
        self.setMinimumHeight(120)

    def clear(self):
        self._current_dbm = np.nan
        self._hold_dbm = np.nan
        self._history_dbm = []
        self._history_bias = []
        self._pending_point_bias = 0.0
        self.update()

    def set_max_hold_enabled(self, enabled: bool):
        self._max_hold_enabled = bool(enabled)
        if self._max_hold_enabled and np.isfinite(self._current_dbm):
            self._hold_dbm = float(self._current_dbm)
        elif not self._max_hold_enabled:
            self._hold_dbm = np.nan
        self.update()

    def set_level_dbm(self, level_dbm: float):
        try:
            level = float(level_dbm)
        except Exception:
            self.clear()
            return

        if np.isnan(level):
            self.clear()
            return

        level = float(np.clip(level, DBM_MIN, DBM_MAX))
        self._current_dbm = level

        if not self._history_dbm:
            self._history_dbm = [level] * int(self._history_capacity)
            self._history_bias = [0.0] * int(self._history_capacity)
        else:
            prev_level = float(self._history_dbm[-1])
            prev_bias = float(self._history_bias[-1]) if self._history_bias else 0.0
            seg_count = max(2, int(self._segment_points))
            t_values = np.linspace(1.0 / seg_count, 1.0, num=seg_count, dtype=np.float64)
            delta = level - prev_level
            curve_dbm = float(np.clip(delta * 0.18, -2.2, 2.2))
            ripple_dbm = float(self._pending_point_bias * 1.0)

            new_levels = []
            new_biases = []
            for t in t_values:
                ease = 0.5 - 0.5 * np.cos(np.pi * t)
                seg_bias = ((1.0 - t) * prev_bias) + (t * self._pending_point_bias)
                shaped = (
                    prev_level
                    + (delta * ease)
                    + (np.sin(np.pi * t) * curve_dbm)
                    + (np.sin(2.0 * np.pi * t) * ripple_dbm)
                )
                new_levels.append(float(np.clip(shaped, DBM_MIN, DBM_MAX)))
                new_biases.append(float(seg_bias))

            self._history_dbm.extend(new_levels)
            self._history_bias.extend(new_biases)
            if len(self._history_dbm) > self._history_capacity:
                self._history_dbm = self._history_dbm[-self._history_capacity :]
                self._history_bias = self._history_bias[-self._history_capacity :]

        if self._max_hold_enabled:
            if np.isnan(self._hold_dbm):
                self._hold_dbm = level
            else:
                self._hold_dbm = max(float(self._hold_dbm), level)
        self.update()

    def push_spectrum_dbm(self, spectrum_dbm):
        try:
            arr = np.asarray(spectrum_dbm, dtype=np.float32).reshape(-1)
        except Exception:
            self._pending_point_bias = 0.0
            return

        if arr.size <= 0 or np.all(np.isnan(arr)):
            self._pending_point_bias = 0.0
            return

        valid = np.isfinite(arr)
        if not np.any(valid):
            self._pending_point_bias = 0.0
            return

        if not np.all(valid):
            finite_vals = arr[valid]
            fill_value = float(np.mean(finite_vals)) if finite_vals.size else float(DBM_MIN)
            arr = np.where(valid, arr, fill_value)

        centered = arr - float(np.mean(arr))
        if centered.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= float(np.sum(kernel))
            centered = np.convolve(centered, kernel, mode="same").astype(np.float32)

        if centered.size >= 2:
            slope = float(centered[-1] - centered[0])
        else:
            slope = 0.0
        roughness = float(np.std(centered)) if centered.size else 0.0
        span = float(np.percentile(np.abs(centered), 90)) if centered.size else 0.0
        if span <= 1e-4 or roughness <= 1e-4:
            self._pending_point_bias = 0.0
            return

        bias = np.clip((slope / span) * 0.9, -1.0, 1.0)
        self._pending_point_bias = float(bias)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        r = self.rect()
        p.fillRect(r, QtGui.QColor("#101010"))

        pen_border = QtGui.QPen(QtGui.QColor("#303030"))
        p.setPen(pen_border)
        p.drawRect(r.adjusted(0, 0, -1, -1))

        title_rect = r.adjusted(8, 4, -8, -4)
        p.setPen(QtGui.QColor("#aaaaaa"))
        p.drawText(title_rect, Qt.AlignLeft | Qt.AlignTop, "Horizontal dBm")

        if np.isnan(self._current_dbm) or not self._history_dbm:
            p.setPen(QtGui.QColor("#aaaaaa"))
            p.drawText(r.adjusted(8, 4, -8, -4), Qt.AlignCenter, "No data")
            return

        left = r.left() + 42
        top = r.top() + 24
        right = r.right() - 12
        bottom = r.bottom() - 18

        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)

        plot_rect = QtCore.QRectF(left, top, plot_w, plot_h)
        p.fillRect(plot_rect, QtGui.QColor("#13171d"))

        grid_pen = QtGui.QPen(QtGui.QColor("#27303a"))
        grid_pen.setStyle(Qt.DashLine)
        grid_pen.setDashPattern([3.0, 4.0])
        p.setPen(grid_pen)
        for frac in (0.25, 0.50, 0.75):
            y = top + (plot_h * frac)
            p.drawLine(QtCore.QPointF(left, y), QtCore.QPointF(right, y))

        def _dbm_to_y(dbm_value: float) -> float:
            ratio = (float(dbm_value) - DBM_MIN) / float(DBM_MAX - DBM_MIN)
            ratio = max(0.0, min(1.0, ratio))
            return float(bottom - (ratio * plot_h))

        history = np.asarray(self._history_dbm, dtype=np.float64)
        history_bias = np.asarray(self._history_bias, dtype=np.float64)
        if history.size == 1:
            history = np.repeat(history, 2)
            history_bias = np.repeat(history_bias, 2)

        sample_count = max(2, int(self._history_capacity))
        slot_x = np.linspace(left, right, num=sample_count, dtype=np.float64)
        x_coords = slot_x[-history.size :]
        y_coords = np.array([_dbm_to_y(v) for v in history], dtype=np.float64)
        if history_bias.size == history.size:
            amp_px = max(1.5, min(plot_h * 0.035, 5.0))
            y_coords = np.clip(y_coords - (history_bias * amp_px), top + 2.0, bottom - 2.0)

        trace_path = QtGui.QPainterPath()
        trace_path.moveTo(float(x_coords[0]), float(y_coords[0]))
        for x, y in zip(x_coords[1:], y_coords[1:]):
            trace_path.lineTo(float(x), float(y))

        fill_path = QtGui.QPainterPath()
        fill_path.moveTo(float(x_coords[0]), float(bottom))
        for x, y in zip(x_coords, y_coords):
            fill_path.lineTo(float(x), float(y))
        fill_path.lineTo(float(x_coords[-1]), float(bottom))
        fill_path.closeSubpath()

        fill_grad = QtGui.QLinearGradient(left, top, left, bottom)
        fill_grad.setColorAt(0.0, QtGui.QColor(44, 243, 178, 72))
        fill_grad.setColorAt(1.0, QtGui.QColor(44, 243, 178, 10))
        p.fillPath(fill_path, QtGui.QBrush(fill_grad))

        current_pen = QtGui.QPen(QtGui.QColor("#2cf3b2"))
        current_pen.setWidth(3)
        p.setPen(current_pen)
        p.drawPath(trace_path)

        current_y = float(y_coords[-1])

        if self._max_hold_enabled and np.isfinite(self._hold_dbm):
            hold_y = _dbm_to_y(self._hold_dbm)
            hold_pen = QtGui.QPen(QtGui.QColor("#ffb347"))
            hold_pen.setWidth(2)
            hold_pen.setStyle(Qt.DashLine)
            p.setPen(hold_pen)
            p.drawLine(QtCore.QPointF(left, hold_y), QtCore.QPointF(right, hold_y))

        marker_pen = QtGui.QPen(QtGui.QColor("#e6fff8"))
        marker_pen.setWidth(1)
        p.setPen(marker_pen)
        p.drawLine(QtCore.QPointF(left - 8, current_y), QtCore.QPointF(left, current_y))
        p.drawLine(QtCore.QPointF(right, current_y), QtCore.QPointF(right + 8, current_y))
        p.drawEllipse(QtCore.QPointF(float(x_coords[-1]), current_y), 3.0, 3.0)

        p.setPen(QtGui.QColor("#888888"))
        p.drawText(
            QtCore.QRectF(4, top - 4, left - 10, 16),
            Qt.AlignRight | Qt.AlignTop,
            f"{DBM_MAX:.0f}",
        )
        p.drawText(
            QtCore.QRectF(4, top + (plot_h * 0.5) - 8, left - 10, 16),
            Qt.AlignRight | Qt.AlignTop,
            f"{(DBM_MIN + DBM_MAX) / 2.0:.0f}",
        )
        p.drawText(
            QtCore.QRectF(4, bottom - 10, left - 10, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{DBM_MIN:.0f}",
        )

        stats_text = f"Current {self._current_dbm:0.1f} dBm"
        if self._max_hold_enabled and np.isfinite(self._hold_dbm):
            stats_text += f"   Hold {self._hold_dbm:0.1f} dBm"
        p.setPen(QtGui.QColor("#9db4c8"))
        p.drawText(
            QtCore.QRectF(left, r.top() + 4, plot_w, 16),
            Qt.AlignRight | Qt.AlignTop,
            stats_text,
        )


class SpectrumTraceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trace_dbm = None
        self._hold_trace_dbm = None
        self._tracker_info = None
        self._max_hold_enabled = False
        self.setMinimumHeight(220)

    def clear(self):
        self._trace_dbm = None
        self._hold_trace_dbm = None
        self._tracker_info = None
        self.update()

    def set_max_hold_enabled(self, enabled: bool):
        enabled = bool(enabled)
        self._max_hold_enabled = enabled
        if not enabled:
            self._hold_trace_dbm = None
        elif self._trace_dbm is not None:
            self._hold_trace_dbm = np.array(self._trace_dbm, copy=True)
        self.update()

    def push_spectrum_dbm(self, spectrum_dbm):
        try:
            arr = np.asarray(spectrum_dbm, dtype=np.float32).reshape(-1)
        except Exception:
            self.clear()
            return

        if arr.size <= 0 or np.all(np.isnan(arr)):
            self.clear()
            return

        valid = np.isfinite(arr)
        if not np.any(valid):
            self.clear()
            return

        if not np.all(valid):
            finite_vals = arr[valid]
            fill_value = float(np.mean(finite_vals)) if finite_vals.size else float(DBM_MIN)
            arr = np.where(valid, arr, fill_value)

        if arr.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= float(np.sum(kernel))
            arr = np.convolve(arr, kernel, mode="same").astype(np.float32)

        self._trace_dbm = arr
        if self._max_hold_enabled:
            if self._hold_trace_dbm is None or self._hold_trace_dbm.shape != arr.shape:
                self._hold_trace_dbm = np.array(arr, copy=True)
            else:
                self._hold_trace_dbm = np.maximum(self._hold_trace_dbm, arr)
        self.update()

    def set_tracker_info(self, tracker_info):
        self._tracker_info = dict(tracker_info) if isinstance(tracker_info, dict) else None
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        r = self.rect()
        p.fillRect(r, QtGui.QColor("#0f1115"))
        p.setPen(QtGui.QPen(QtGui.QColor("#30343b")))
        p.drawRect(r.adjusted(0, 0, -1, -1))

        title_rect = r.adjusted(10, 6, -10, -6)
        p.setPen(QtGui.QColor("#c6d4e1"))
        p.drawText(title_rect, Qt.AlignLeft | Qt.AlignTop, "Live Spectrum")

        left = r.left() + 42
        top = r.top() + 24
        right = r.right() - 12
        bottom = r.bottom() - 22
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)
        plot_rect = QtCore.QRectF(left, top, plot_w, plot_h)
        p.fillRect(plot_rect, QtGui.QColor("#131922"))

        grid_pen = QtGui.QPen(QtGui.QColor("#27303a"))
        grid_pen.setStyle(Qt.DashLine)
        grid_pen.setDashPattern([3.0, 4.0])
        p.setPen(grid_pen)
        for frac in (0.20, 0.40, 0.60, 0.80):
            y = top + (plot_h * frac)
            p.drawLine(QtCore.QPointF(left, y), QtCore.QPointF(right, y))
        for frac in (0.25, 0.50, 0.75):
            x = left + (plot_w * frac)
            p.drawLine(QtCore.QPointF(x, top), QtCore.QPointF(x, bottom))

        def _dbm_to_y(dbm_value: float) -> float:
            ratio = (float(dbm_value) - DBM_MIN) / float(DBM_MAX - DBM_MIN)
            ratio = max(0.0, min(1.0, ratio))
            return float(bottom - (ratio * plot_h))

        if self._trace_dbm is None or self._trace_dbm.size < 2:
            p.setPen(QtGui.QColor("#8a93a1"))
            p.drawText(plot_rect.toRect(), Qt.AlignCenter, "No spectrum data")
            return

        trace = self._trace_dbm.astype(np.float64)
        x_coords = np.linspace(left, right, num=trace.size, dtype=np.float64)
        y_coords = np.array([_dbm_to_y(v) for v in trace], dtype=np.float64)

        if self._hold_trace_dbm is not None and self._hold_trace_dbm.shape == self._trace_dbm.shape:
            hold = self._hold_trace_dbm.astype(np.float64)
            hold_y = np.array([_dbm_to_y(v) for v in hold], dtype=np.float64)
            hold_path = QtGui.QPainterPath()
            hold_path.moveTo(float(x_coords[0]), float(hold_y[0]))
            for x, y in zip(x_coords[1:], hold_y[1:]):
                hold_path.lineTo(float(x), float(y))
            hold_pen = QtGui.QPen(QtGui.QColor("#ffb347"))
            hold_pen.setWidth(2)
            hold_pen.setStyle(Qt.DashLine)
            p.setPen(hold_pen)
            p.drawPath(hold_path)

        fill_path = QtGui.QPainterPath()
        fill_path.moveTo(float(x_coords[0]), float(bottom))
        for x, y in zip(x_coords, y_coords):
            fill_path.lineTo(float(x), float(y))
        fill_path.lineTo(float(x_coords[-1]), float(bottom))
        fill_path.closeSubpath()

        fill_grad = QtGui.QLinearGradient(left, top, left, bottom)
        fill_grad.setColorAt(0.0, QtGui.QColor(86, 182, 255, 96))
        fill_grad.setColorAt(1.0, QtGui.QColor(86, 182, 255, 12))
        p.fillPath(fill_path, QtGui.QBrush(fill_grad))

        trace_path = QtGui.QPainterPath()
        trace_path.moveTo(float(x_coords[0]), float(y_coords[0]))
        for x, y in zip(x_coords[1:], y_coords[1:]):
            trace_path.lineTo(float(x), float(y))
        trace_pen = QtGui.QPen(QtGui.QColor("#7bd0ff"))
        trace_pen.setWidth(2)
        p.setPen(trace_pen)
        p.drawPath(trace_path)

        tracker_info = self._tracker_info if isinstance(self._tracker_info, dict) else None
        if tracker_info and int(tracker_info.get("fft_size", 0)) == int(trace.size):
            peak_bin = int(np.clip(int(tracker_info.get("peak_bin", 0)), 0, trace.size - 1))
            track_x = float(x_coords[peak_bin])
            track_y = float(y_coords[peak_bin])
            marker_pen = QtGui.QPen(QtGui.QColor("#ffd166"))
            marker_pen.setWidth(2)
            marker_pen.setStyle(Qt.DashLine)
            p.setPen(marker_pen)
            p.drawLine(QtCore.QPointF(track_x, top), QtCore.QPointF(track_x, bottom))
            p.setBrush(QtGui.QBrush(QtGui.QColor("#ffd166")))
            p.drawEllipse(QtCore.QPointF(track_x, track_y), 4.0, 4.0)

        center_x = left + (plot_w * 0.5)
        center_pen = QtGui.QPen(QtGui.QColor("#4a5563"))
        center_pen.setStyle(Qt.DotLine)
        p.setPen(center_pen)
        p.drawLine(QtCore.QPointF(center_x, top), QtCore.QPointF(center_x, bottom))

        p.setPen(QtGui.QColor("#8d99a8"))
        p.drawText(QtCore.QRectF(4, top - 6, left - 10, 16), Qt.AlignRight | Qt.AlignTop, f"{DBM_MAX:.0f}")
        p.drawText(
            QtCore.QRectF(4, top + (plot_h * 0.5) - 8, left - 10, 16),
            Qt.AlignRight | Qt.AlignTop,
            f"{(DBM_MIN + DBM_MAX) / 2.0:.0f}",
        )
        p.drawText(QtCore.QRectF(4, bottom - 10, left - 10, 16), Qt.AlignRight | Qt.AlignVCenter, f"{DBM_MIN:.0f}")

        peak_dbm = float(np.max(trace))
        floor_dbm = float(np.min(trace))
        if tracker_info and np.isfinite(float(tracker_info.get("tracked_dbm", np.nan))):
            tracked_dbm = float(tracker_info.get("tracked_dbm", np.nan))
            tracked_freq_hz = tracker_info.get("tracked_freq_hz", None)
            tracked_freq_text = ""
            if tracked_freq_hz is not None and np.isfinite(float(tracked_freq_hz)):
                tracked_freq_text = f"   {float(tracked_freq_hz) / 1e6:0.4f} MHz"
            stats = (
                f"Track {tracked_dbm:0.1f} dBm{tracked_freq_text}"
                f"   Peak {peak_dbm:0.1f} dBm   Floor {floor_dbm:0.1f} dBm"
            )
        else:
            stats = f"Peak {peak_dbm:0.1f} dBm   Floor {floor_dbm:0.1f} dBm   Bins {trace.size}"
        p.setPen(QtGui.QColor("#a7c6df"))
        p.drawText(QtCore.QRectF(left, r.top() + 4, plot_w, 16), Qt.AlignRight | Qt.AlignTop, stats)


class SpectrumDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum")
        self.resize(820, 360)
        self.setModal(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.spectrum_widget = SpectrumTraceWidget(self)
        layout.addWidget(self.spectrum_widget, 1)


class ContinuousToneDevice(QtCore.QIODevice):
    def __init__(self, sample_rate: int = 22050, parent=None):
        super().__init__(parent)
        self.sample_rate = max(8000, int(sample_rate))
        self.bytes_per_frame = 2  # 16-bit mono PCM
        self._phase = 0.0
        self._mod_phase = 0.0
        self._current_freq_hz = 280.0
        self._target_freq_hz = 280.0
        self._current_gain = 0.0
        self._target_gain = 0.0
        self._pitch_smooth_s = 0.045
        self._gain_attack_s = 0.025
        self._gain_release_s = 0.070
        self._tremolo_rate_hz = 1.35
        self._tremolo_depth = 0.32

    def reset_state(self):
        self._phase = 0.0
        self._mod_phase = 0.0
        self._current_freq_hz = float(self._target_freq_hz)
        self._current_gain = 0.0
        self._target_gain = 0.0

    def set_tone(self, frequency_hz: float, gain: float):
        freq = float(np.clip(frequency_hz, 260.0, 900.0))
        vol = float(np.clip(gain, 0.0, 0.40))
        self._target_freq_hz = freq
        self._target_gain = vol
        if self._current_freq_hz <= 0.0:
            self._current_freq_hz = freq

    def isSequential(self):
        return True

    def bytesAvailable(self):
        return 8192 + super().bytesAvailable()

    def readData(self, maxlen: int):
        if maxlen <= 0:
            return b""

        frame_count = max(1, maxlen // self.bytes_per_frame)

        if self._target_gain <= 1e-4 and self._current_gain <= 1e-4:
            self._current_gain = 0.0
            self._current_freq_hz = float(self._target_freq_hz)
            return bytes(frame_count * self.bytes_per_frame)

        pitch_alpha = min(
            1.0,
            frame_count / max(1.0, self.sample_rate * self._pitch_smooth_s),
        )
        gain_smooth_s = (
            self._gain_attack_s
            if self._target_gain >= self._current_gain
            else self._gain_release_s
        )
        gain_alpha = min(
            1.0,
            frame_count / max(1.0, self.sample_rate * gain_smooth_s),
        )

        next_freq = self._current_freq_hz + (
            (self._target_freq_hz - self._current_freq_hz) * pitch_alpha
        )
        next_gain = self._current_gain + (
            (self._target_gain - self._current_gain) * gain_alpha
        )

        freq_curve = np.linspace(
            self._current_freq_hz,
            next_freq,
            num=frame_count,
            endpoint=False,
            dtype=np.float64,
        )
        gain_curve = np.linspace(
            self._current_gain,
            next_gain,
            num=frame_count,
            endpoint=False,
            dtype=np.float64,
        )

        phase_step = (2.0 * np.pi * freq_curve) / float(self.sample_rate)
        phases = self._phase + np.cumsum(phase_step)
        mod_step = (2.0 * np.pi * self._tremolo_rate_hz) / float(self.sample_rate)
        mod_phases = self._mod_phase + (np.arange(frame_count, dtype=np.float64) * mod_step)
        tremolo = 1.0 - self._tremolo_depth + (
            self._tremolo_depth * (0.5 * (1.0 + np.sin(mod_phases)))
        )
        samples = np.sin(phases) * gain_curve * tremolo
        pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)

        self._phase = float(phases[-1] % (2.0 * np.pi))
        self._mod_phase = float((mod_phases[-1] + mod_step) % (2.0 * np.pi))
        self._current_freq_hz = float(next_freq)
        self._current_gain = float(next_gain)
        return pcm.tobytes()

    def writeData(self, data):
        return 0


class ContinuousTonePlayer(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_output = None
        self.tone_device = None
        self._running = False
        self._available = False
        self._init_output()

    def _init_output(self):
        if not hasattr(QtMultimedia, "QAudioOutput"):
            print("[TONE] QAudioOutput is not available on this build")
            return

        try:
            audio_format = QtMultimedia.QAudioFormat()
            audio_format.setSampleRate(22050)
            audio_format.setChannelCount(1)
            audio_format.setSampleSize(16)
            audio_format.setCodec("audio/pcm")
            audio_format.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
            audio_format.setSampleType(QtMultimedia.QAudioFormat.SignedInt)

            device_info = QtMultimedia.QAudioDeviceInfo.defaultOutputDevice()
            if device_info.isNull():
                print("[TONE] no default audio output device")
                return

            if not device_info.isFormatSupported(audio_format):
                audio_format = device_info.nearestFormat(audio_format)

            self.tone_device = ContinuousToneDevice(
                sample_rate=audio_format.sampleRate(),
                parent=self,
            )
            self.tone_device.open(QtCore.QIODevice.ReadOnly)

            self.audio_output = QtMultimedia.QAudioOutput(
                device_info,
                audio_format,
                self,
            )
            self.audio_output.setVolume(1.0)
            self._available = True
            print(
                "[TONE] continuous tone ready:",
                audio_format.sampleRate(),
                "Hz",
            )
        except Exception as exc:
            print("[TONE] init failed:", exc)
            self.audio_output = None
            self.tone_device = None
            self._available = False

    def is_available(self) -> bool:
        return bool(self._available and self.audio_output and self.tone_device)

    def start(self):
        if not self.is_available() or self._running:
            return
        try:
            self.audio_output.start(self.tone_device)
            self._running = True
        except Exception as exc:
            print("[TONE] start failed:", exc)

    def stop(self):
        if not self.is_available():
            return
        try:
            self.audio_output.stop()
        except Exception as exc:
            print("[TONE] stop failed:", exc)
        self._running = False
        self.tone_device.reset_state()

    def set_tone(self, frequency_hz: float, gain: float):
        if not self.is_available():
            return
        if not self._running:
            self.start()
        self.tone_device.set_tone(frequency_hz, gain)


class NumericKeypadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_text=""):
        super().__init__(parent)

        # Reuse parent UI scale if available (touch targets stay usable on 480x800).
        self.ui_scale = float(getattr(parent, "ui_scale", 1.0) or 1.0)

        def s(v: float) -> int:
            return int(round(v * self.ui_scale))

        self.setWindowTitle("Input Frekuensi")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        layout = QtWidgets.QVBoxLayout(self)

        self.edit = QtWidgets.QLineEdit(self)
        self.edit.setText(initial_text)
        self.edit.setAlignment(QtCore.Qt.AlignRight)
        self.edit.setReadOnly(True)  # hanya lewat tombol
        self.edit.setStyleSheet(f"font-size: {s(24)}px; padding: {s(6)}px;")
        layout.addWidget(self.edit)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        buttons = [
            ("7", 0, 0), ("8", 0, 1), ("9", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("1", 2, 0), ("2", 2, 1), ("3", 2, 2),
            ("0", 3, 0),
            (".", 3, 1),
            ("Del", 3, 2),
        ]

        for text, r, c in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setMinimumSize(s(70), s(60))
            btn.setStyleSheet(f"font-size: {s(22)}px;")
            grid.addWidget(btn, r, c)
            btn.clicked.connect(lambda _, t=text: self._on_button(t))

        # OK / Cancel
        h = QtWidgets.QHBoxLayout()
        layout.addLayout(h)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        for b in (btn_ok, btn_cancel):
            b.setMinimumHeight(s(50))
            b.setStyleSheet(f"font-size: {s(20)}px;")
        h.addWidget(btn_ok)
        h.addWidget(btn_cancel)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _on_button(self, text):
        if text == "Del":
            self.edit.setText(self.edit.text()[:-1])
        elif text == ".":
            if "." not in self.edit.text():
                self.edit.setText(self.edit.text() + ".")
        else:  # digit
            self.edit.setText(self.edit.text() + text)

    def value(self):
        try:
            return float(self.edit.text())
        except Exception:
            return None



class MainWindow(QtWidgets.QMainWindow, QtCore.QObject):
    def __init__(self, scale: float = 1.0, base_size=(800, 480), fullscreen: bool = True):
        super().__init__()

        # faktor skala UI terhadap ukuran desain dasar
        self.ui_scale = scale
        self.base_size = (int(base_size[0]), int(base_size[1]))
        self.is_portrait = self.base_size[1] >= self.base_size[0]
        self.fullscreen = bool(fullscreen)

        self.setWindowIcon(QtGui.QIcon(str(IMG_DIR / "wifi_icon2.ico")))
        self.setWindowTitle("Waltech Sigmavex")
        self.resize(self.base_size[0], self.base_size[1])

        self.sdr = None
        self.worker = None
        self.worker_thread = None
        self.ble_server = None
        self.ble_enabled = False
        self.ble_local_name = "WIN0001"
        self.ble_client_connected = False
        self.ble_client_detail = "Bluetooth is starting."
        self.measurement_running = False

        self.smoothed_power_dbm = np.nan
        self.audio_power_dbm = np.nan
        self.display_power_dbm = np.nan
        self.last_power_display_ts = 0.0
        self.target_freq_hz = None
        self._latest_spectrum_dbm = None
        self.spectrum_tracker_info = None
        self._ble_spec_str = None  # compact spectrum for BLE status (e.g. "x;x;...")
        self._ble_last_status_signature = None
        self._ble_last_push_ts = 0.0
        self.battery_pct = None
        self.battery_status = None
        self.battery_mv = None
        self.battery_ma = None
        self._ina219_bus = None
        self.bt_audio_devices = []
        self.bt_audio_sink_id = None
        self.bt_audio_connected_mac = ""
        self.bt_audio_connected_name = ""
        self._last_tone_freq_hz = 280.0
        self._last_tone_volume = 0.0
        self.spectrum_mode_enabled = False
        self.spectrum_view_mode = "history"

        # Spectrum UI defaults (applied to worker on start).
        self.spectrum_fft_size = 2048
        self.spectrum_averages = 3
        self.spectrum_interval_s = 0.25
        self.spectrum_max_hold = False
        self.tdd_uplink_assist_enabled = bool(TDD_UPLINK_ASSIST_ENABLED)
        self.tdd_hint_peak_dbm = np.nan
        self.tdd_hint_steady_dbm = np.nan
        self.tdd_hint_burstiness_db = np.nan
        self.tdd_hint_display_state = ""
        self.tdd_hint_hold_until = 0.0

        self._build_ui()
        self._setup_numeric_popups()
        self._init_sdr()
        self._setup_ble_status_timer()

        if self.fullscreen:
            # Fullscreen + tanpa border
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.showFullScreen()
        else:
            self.show()

        # Event filter untuk popup keypad numerik dipasang di _setup_numeric_popups().

        # Tone kontinu untuk panduan arah sinyal.
        self.tone_player = None
        self.beep = None
        self._init_guidance_audio_backend()

        # Kalau startup gagal deteksi SDR, tawarkan reconnect
        while self.sdr is None:
            if not self._prompt_sdr_reconnect():
                break
            self._init_sdr()

        # BLE is always-on; start after the Qt event loop is running.
        QtCore.QTimer.singleShot(0, lambda: self._ble_set_enabled(True))
        QtCore.QTimer.singleShot(1200, self._bt_audio_refresh_devices)

        # Battery indicator (poll sysfs periodically; cheap + robust).
        self._battery_timer = QtCore.QTimer(self)
        self._battery_timer.setInterval(5000)
        self._battery_timer.timeout.connect(self._update_battery_indicator)
        self._battery_timer.start()
        QtCore.QTimer.singleShot(0, self._update_battery_indicator)

    # helper untuk scaling pixel
    def s(self, value: float) -> int:
        return int(round(value * self.ui_scale))

    def _setup_numeric_popups(self):
        # Map widget/lineEdit -> (spinbox, kind)
        # kind: "int" or "float"
        self._numeric_popup_targets = {}

        def add_target(spinbox, kind: str):
            spinbox.setKeyboardTracking(False)
            spinbox.installEventFilter(self)
            self._numeric_popup_targets[spinbox] = (spinbox, kind)

            le = spinbox.lineEdit()
            if le is not None:
                le.setReadOnly(True)  # jangan edit langsung via keyboard OS
                le.installEventFilter(self)
                self._numeric_popup_targets[le] = (spinbox, kind)

        add_target(self.chan_spin, "int")
        add_target(self.gain_spin, "int")
        add_target(self.freq_spin, "float")

    # ---------------------------------------------------------
    # SDR INIT & POPUP
    # ---------------------------------------------------------
    def _init_sdr(self):
        """Init SDR saat startup atau saat diminta ulang."""
        try:
            gain_db = self.gain_spin.value()
            self.sdr = BladeRFPowerMeter(gain=gain_db)

            # Tune awal pakai nilai dari UI
            self.target_freq_hz = self._compute_freq_from_ui_hz()
            self.sdr.tune(self.target_freq_hz)
            self._apply_measurement_profile_to_sdr()

            print("[INIT] SDR initialized")

            # Update status sukses
            self.lbl_status.setText("Ready - Connected")
            self.lbl_status.setStyleSheet(
                f"font-size:{self.s(14)}px; color: green; font-weight:bold;"
            )
            self.btn_start.setEnabled(True)

        except Exception as e:
            print("[Device not found]", e)
            self.sdr = None

            # Status gagal
            self.lbl_status.setText("Device not found - Disconnected")
            self.lbl_status.setStyleSheet(
                f"font-size:{self.s(14)}px; color: red; font-weight:bold;"
            )
            self.btn_start.setEnabled(False)

    def _prompt_sdr_reconnect(self) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Disconnected")
        box.setText(
            "<span style='color: red; font-weight:bold; font-size:15px;'>"
            "Device not found - check connection"
            "</span>"
        )
        reconnect_btn = box.addButton("Reconnect", QtWidgets.QMessageBox.AcceptRole)
        cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(reconnect_btn)

        box.exec_()
        return box.clickedButton() == reconnect_btn

    def _setup_ble_status_timer(self):
        self._ble_status_timer = QtCore.QTimer(self)
        self._ble_status_timer.setInterval(400)
        self._ble_status_timer.timeout.connect(self._ble_push_status_if_enabled)
        self._ble_status_timer.start()

    def _ble_log_line(self, line: str):
        try:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {str(line)}")
        except Exception:
            pass

    def _set_ble_connection_indicator(self, connected: bool, detail: str = ""):
        self.ble_client_connected = bool(connected)
        self.ble_client_detail = str(detail or "").strip()

        if not hasattr(self, "lbl_ble_phone_state"):
            return

        detail_text = self.ble_client_detail
        detail_lc = detail_text.lower()
        if self.ble_client_connected:
            state_text = "Connected to phone"
            if not detail_text:
                detail_text = "Android device is connected to this BLE service."
            frame_bg = "#14261a"
            frame_border = "#2e7d32"
            dot_color = "#39d353"
            state_color = "#d9ffe2"
            detail_color = "#b9d6c1"
        else:
            state_text = "Phone not connected"
            if not detail_text:
                detail_text = "Waiting for Android device connection."
            is_error = any(
                key in detail_lc
                for key in (
                    "failed",
                    "not supported",
                    "not usable",
                    "disabled",
                    "unavailable",
                    "error",
                )
            )
            if is_error:
                frame_bg = "#2c1717"
                frame_border = "#c62828"
                dot_color = "#ef5350"
                state_color = "#ffd9d9"
                detail_color = "#f3b4b4"
            else:
                frame_bg = "#2a2013"
                frame_border = "#ef6c00"
                dot_color = "#ff9800"
                state_color = "#ffe5c2"
                detail_color = "#e9c9a2"

        self.ble_status_card.setStyleSheet(
            f"""
            QFrame#bleStatusCard {{
                background-color: {frame_bg};
                border: 1px solid {frame_border};
                border-radius: {self.s(12)}px;
            }}
            """
        )
        dot_size = self.s(14)
        self.lbl_ble_phone_dot.setStyleSheet(
            f"background-color:{dot_color}; border-radius:{max(1, dot_size // 2)}px;"
        )
        self.lbl_ble_phone_state.setText(state_text)
        self.lbl_ble_phone_state.setStyleSheet(
            f"font-size:{self.s(18)}px; font-weight:bold; color:{state_color};"
        )
        self.lbl_ble_phone_detail.setText(detail_text)
        self.lbl_ble_phone_detail.setStyleSheet(
            f"font-size:{self.s(12)}px; color:{detail_color};"
        )

    @QtCore.pyqtSlot(bool, str)
    def _on_ble_connection_changed(self, connected: bool, detail: str):
        self._set_ble_connection_indicator(bool(connected), str(detail or ""))

    def _ble_is_compact_status_mode(self) -> bool:
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        return bool(channel_mode_uses_shared_tdd_carrier(mode))

    def _ble_min_push_interval_s(self) -> float:
        if self._ble_is_compact_status_mode():
            return 0.75
        return 0.35

    def _reset_ble_status_cache(self):
        self._ble_last_status_signature = None
        self._ble_last_push_ts = 0.0

    def _ble_build_status(self) -> dict:
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        gain = int(self.gain_spin.value()) if hasattr(self, "gain_spin") else 0
        running = self.worker_thread is not None
        compact_mode = self._ble_is_compact_status_mode()

        target_mhz = None
        if self.target_freq_hz is not None:
            try:
                target_mhz = float(self.target_freq_hz) / 1e6
            except Exception:
                target_mhz = None

        dbm = None
        dbm_source = self.smoothed_power_dbm
        if compact_mode and not np.isnan(self.display_power_dbm):
            dbm_source = self.display_power_dbm
        if not np.isnan(dbm_source):
            dbm = float(dbm_source)

        st = {
            "ok": True,
            "ts": int(time.time() * 1000),
            "mode": mode,
            "shared_tdd_carrier": bool(channel_mode_uses_shared_tdd_carrier(mode)),
            "gain_db": gain,
            "running": bool(running),
            "sdr_connected": bool(self.sdr is not None),
            "target_freq_mhz": target_mhz,
            "dbm": dbm,
            "status_text": self.lbl_status.text() if hasattr(self, "lbl_status") else "",
        }

        if compact_mode:
            hint_map = {
                "uplink_burst": "ul",
                "downlink_steady": "dl",
                "mixed": "mix",
            }
            hint = hint_map.get(str(getattr(self, "tdd_hint_display_state", "")).strip())
            if hint:
                st["tdd_hint"] = hint
            return st

        st["measurement_note"] = channel_mode_measurement_note(mode)
        spec = getattr(self, "_ble_spec_str", None)
        if spec:
            st["spec_n"] = len(str(spec).split(";"))
            st["spec"] = str(spec)
        if self.battery_pct is not None:
            st["battery_pct"] = int(self.battery_pct)
        if self.battery_status:
            st["battery_status"] = str(self.battery_status)
        if self.battery_mv is not None:
            st["battery_mv"] = int(self.battery_mv)
        if self.battery_ma is not None:
            st["battery_ma"] = int(self.battery_ma)
        return st

    def _read_battery_sysfs(self):
        """
        Best-effort battery read from Linux sysfs:
        /sys/class/power_supply/<name>/{type,capacity,status,voltage_now,...}
        Returns dict or None.
        """
        base = Path("/sys/class/power_supply")
        if not base.exists():
            return None
        try:
            for dev in base.iterdir():
                try:
                    t = (dev / "type").read_text().strip()
                except Exception:
                    continue
                if t.lower() != "battery":
                    continue

                def _read_int(p: Path):
                    try:
                        return int(p.read_text().strip())
                    except Exception:
                        return None

                def _read_str(p: Path):
                    try:
                        return p.read_text().strip()
                    except Exception:
                        return None

                cap = _read_int(dev / "capacity")
                if cap is None:
                    # Fallback: energy/charge ratio.
                    now = _read_int(dev / "energy_now") or _read_int(dev / "charge_now")
                    full = _read_int(dev / "energy_full") or _read_int(dev / "charge_full")
                    if now is not None and full:
                        cap = int(round(100.0 * float(now) / float(full)))

                status = _read_str(dev / "status")
                v_now = _read_int(dev / "voltage_now")
                # voltage_now is usually in microvolts
                mv = int(round(v_now / 1000.0)) if v_now is not None else None

                if cap is None and status is None and mv is None:
                    continue

                if cap is not None:
                    cap = 0 if cap < 0 else (100 if cap > 100 else cap)

                return {"pct": cap, "status": status, "mv": mv, "name": dev.name}
        except Exception:
            return None
        return None

    def _estimate_battery_pct_from_voltage(self, mv):
        try:
            mv = int(mv)
        except Exception:
            return None

        empty_mv = int(BATTERY_VOLTAGE_EMPTY_MV)
        full_mv = int(BATTERY_VOLTAGE_FULL_MV)
        if full_mv <= empty_mv:
            return None

        pct = int(round((float(mv) - empty_mv) * 100.0 / float(full_mv - empty_mv)))
        if pct < 0:
            return 0
        if pct > 100:
            return 100
        return pct

    def _get_ina219_bus(self):
        if smbus is None or not sys.platform.startswith("linux"):
            return None

        if self._ina219_bus is not None:
            return self._ina219_bus

        try:
            self._ina219_bus = smbus.SMBus(int(INA219_I2C_BUS_ID))
        except Exception:
            self._ina219_bus = None
        return self._ina219_bus

    def _ina219_read_u16(self, bus, reg: int) -> int:
        data = bus.read_i2c_block_data(int(INA219_I2C_ADDR), int(reg), 2)
        if len(data) != 2:
            raise OSError("INA219 short read")
        return ((int(data[0]) & 0xFF) << 8) | (int(data[1]) & 0xFF)

    def _ina219_read_s16(self, bus, reg: int) -> int:
        raw = self._ina219_read_u16(bus, reg)
        if raw & 0x8000:
            raw -= 0x10000
        return raw

    def _read_battery_ina219(self):
        """
        Fallback battery read from INA219.
        Uses bus voltage + shunt voltage to estimate source voltage.
        """
        bus = self._get_ina219_bus()
        if bus is None:
            return None

        try:
            bus_raw = self._ina219_read_u16(bus, 0x02)
            shunt_raw = self._ina219_read_s16(bus, 0x01)
        except Exception:
            return None

        bus_mv = int(((bus_raw >> 3) & 0x1FFF) * 4)
        shunt_uv = int(shunt_raw * 10)
        source_mv = int(round(bus_mv + (shunt_uv / 1000.0)))
        if source_mv <= 0:
            source_mv = bus_mv
        if source_mv <= 0:
            return None

        shunt_ohms = float(INA219_SHUNT_OHMS) if float(INA219_SHUNT_OHMS) > 0.0 else 0.1
        current_ma = shunt_uv / (1000.0 * shunt_ohms)
        current_ma = int(round(current_ma))

        status = None
        idle_threshold_ma = abs(float(INA219_CURRENT_IDLE_THRESHOLD_MA))
        if current_ma > idle_threshold_ma:
            status = "Discharging"
        elif current_ma < -idle_threshold_ma:
            status = "Charging"
        else:
            status = "Idle"

        return {
            "pct": self._estimate_battery_pct_from_voltage(source_mv),
            "status": status,
            "mv": source_mv,
            "ma": current_ma,
            "name": "ina219",
        }

    def _merge_battery_info(self, primary, fallback):
        if not primary:
            return fallback
        if not fallback:
            return primary

        merged = dict(primary)
        for key, value in fallback.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
        return merged

    @QtCore.pyqtSlot()
    def _update_battery_indicator(self):
        info = self._merge_battery_info(
            self._read_battery_sysfs(),
            self._read_battery_ina219(),
        )
        if not info:
            self.battery_pct = None
            self.battery_status = None
            self.battery_mv = None
            self.battery_ma = None
            try:
                if hasattr(self, "lbl_battery"):
                    self.lbl_battery.setText("Battery: N/A")
            except Exception:
                pass
            return

        self.battery_pct = info.get("pct")
        self.battery_status = info.get("status")
        self.battery_mv = info.get("mv")
        self.battery_ma = info.get("ma")

        try:
            if hasattr(self, "lbl_battery"):
                pct = self.battery_pct
                st = self.battery_status
                mv = self.battery_mv
                ma = self.battery_ma
                bits = []
                if pct is not None:
                    bits.append(f"{int(pct)}%")
                if st:
                    bits.append(str(st))
                if mv is not None:
                    bits.append(f"{int(mv)} mV")
                if ma is not None:
                    bits.append(f"{int(ma)} mA")
                self.lbl_battery.setText("Battery: " + (" | ".join(bits) if bits else "N/A"))
        except Exception:
            pass

    def _ble_push_status_if_enabled(self):
        if not self.ble_enabled or self.ble_server is None:
            return
        try:
            status = self._ble_build_status()
            signature = dict(status)
            signature.pop("ts", None)
            signature_text = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
            now = time.monotonic()

            if (
                self._ble_last_status_signature == signature_text
                and (now - self._ble_last_push_ts) < 2.0
            ):
                return

            if (now - self._ble_last_push_ts) < self._ble_min_push_interval_s():
                return

            self.ble_server.set_status(status)
            self._ble_last_status_signature = signature_text
            self._ble_last_push_ts = now
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def _on_ble_command_json(self, cmd_json: str):
        try:
            cmd = json.loads(cmd_json) if cmd_json else {}
        except Exception:
            return
        if not isinstance(cmd, dict):
            return

        # Supported commands (JSON object):
        # - {"set_mode": "<mode string>"}
        # - {"set_gain_db": 10}
        # - {"set_freq_mhz": 935.2}   (manual mode)
        # - {"set_channel": 2}        (arfcn/earfcn/nrarfcn, depends on mode)
        # - {"apply": true}           (press SET PARA)
        # - {"running": true|false}   (start/stop)
        # - {"power_action": "reboot"|"shutdown"}
        try:
            if "set_mode" in cmd:
                mode = str(cmd.get("set_mode", "")).strip()
                idx = self.mode_combo.findText(mode)
                if idx >= 0:
                    self.mode_combo.setCurrentIndex(idx)

            if "set_gain_db" in cmd:
                try:
                    self.gain_spin.setValue(int(cmd["set_gain_db"]))
                except Exception:
                    pass

            if "set_freq_mhz" in cmd:
                try:
                    self.freq_spin.setValue(float(cmd["set_freq_mhz"]))
                except Exception:
                    pass

            if "set_channel" in cmd:
                try:
                    self.chan_spin.setValue(int(cmd["set_channel"]))
                except Exception:
                    pass

            if cmd.get("apply") is True:
                self._on_set()

            if "running" in cmd:
                desired = bool(cmd.get("running"))
                if desired and self.worker_thread is None:
                    self._on_start()
                elif (not desired) and self.worker_thread is not None:
                    self._on_stop()

            if "power_action" in cmd:
                action = str(cmd.get("power_action", "")).strip().lower()
                if action == "reboot":
                    self._perform_reboot()
                elif action == "shutdown":
                    self._perform_shutdown()
                elif action:
                    raise ValueError(f"unsupported power_action: {action}")
        finally:
            self._reset_ble_status_cache()
            self._ble_push_status_if_enabled()

    def _ble_set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.ble_enabled:
            return
        self.ble_enabled = enabled

        if not enabled:
            if self.ble_server is not None:
                try:
                    self.ble_server.stop()
                except Exception:
                    pass
            self.ble_server = None
            self._reset_ble_status_cache()
            self._set_ble_connection_indicator(False, "Bluetooth is disabled.")
            self._ble_log_line("Bluetooth disabled.")
            return

        if not sys.platform.startswith("linux"):
            self._set_ble_connection_indicator(False, "Bluetooth is not supported on this device.")
            self._ble_log_line("Device not supported.")
            self.ble_enabled = False
            return

        try:
            from ble.bluez_gatt_server import BleServer, BleServerConfig
        except Exception as e:
            self._set_ble_connection_indicator(False, f"Bluetooth module import failed: {e}")
            self._ble_log_line(f"Bluetooth module import failed: {e}")
            self.ble_enabled = False
            return

        def on_command(cmd: dict):
            # Called from BLE thread (GLib); hand off to Qt UI thread.
            try:
                payload = json.dumps(cmd, ensure_ascii=False)
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_on_ble_command_json",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, payload),
                )
            except Exception as e:
                self._ble_log_line(f"Bluetooth failed: {e}")

        def on_connection_changed(connected: bool, detail: str):
            try:
                QtCore.QMetaObject.invokeMethod(
                    self,
                    "_on_ble_connection_changed",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(bool, bool(connected)),
                    QtCore.Q_ARG(str, str(detail or "")),
                )
            except Exception as e:
                self._ble_log_line(f"Bluetooth connection callback failed: {e}")

        cfg = BleServerConfig(local_name=str(self.ble_local_name or "win0001"))
        self.ble_server = BleServer(
            cfg,
            on_command=on_command,
            logger=self._ble_log_line,
            on_connection_changed=on_connection_changed,
        )
        self.ble_server.start()
        self._set_ble_connection_indicator(
            False,
            f"Advertising as '{cfg.local_name}'. Waiting for Android device connection.",
        )
        self._ble_log_line(f"BLE starting: '{cfg.local_name}' (lihat log untuk status register/advertising)")
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    def _init_guidance_audio_backend(self):
        if self.tone_player is not None:
            try:
                self.tone_player.stop()
            except Exception:
                pass
            try:
                self.tone_player.deleteLater()
            except Exception:
                pass
        if self.beep is not None:
            try:
                self.beep.stop()
            except Exception:
                pass
            try:
                self.beep.deleteLater()
            except Exception:
                pass

        self.tone_player = ContinuousTonePlayer(self)
        self.beep = None

        sound_path = str(WAV_DIR / "tone.wav")
        if self.tone_player.is_available():
            print("[TONE] using synthesized continuous tone")
            return

        print("[TONE] fallback WAV path:", sound_path, "exists:", os.path.exists(sound_path))
        self.beep = QtMultimedia.QSoundEffect(self)
        self.beep.setSource(QtCore.QUrl.fromLocalFile(sound_path))
        self.beep.setVolume(0.0)
        self.beep.setLoopCount(QtMultimedia.QSoundEffect.Infinite)
        self.beep.statusChanged.connect(self._on_beep_status_changed)

    def _recreate_guidance_audio_backend(self):
        was_running = bool(self.measurement_running)
        tone_freq = float(getattr(self, "_last_tone_freq_hz", 280.0))
        tone_vol = float(getattr(self, "_last_tone_volume", 0.0))
        self._stop_guidance_tone()
        self._init_guidance_audio_backend()
        if was_running:
            self._start_guidance_tone()
            self._set_guidance_tone(tone_freq, tone_vol)

    def _run_capture_command(self, cmd, timeout_s: float = 12.0, check: bool = True) -> str:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(timeout_s),
                check=False,
            )
        except FileNotFoundError as e:
            raise RuntimeError(str(e))
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Command timed out: {' '.join(cmd)}") from e
        except Exception as e:
            raise RuntimeError(str(e)) from e

        output = (proc.stdout or "").strip()
        error = (proc.stderr or "").strip()
        if check and proc.returncode != 0:
            raise RuntimeError(error or output or f"Command failed: {' '.join(cmd)}")
        return output

    def _run_btctl(self, args, timeout_s: float = 15.0, check: bool = True) -> str:
        return self._run_capture_command(["bluetoothctl", *list(args)], timeout_s=timeout_s, check=check)

    def _run_wpctl(self, args, timeout_s: float = 10.0, check: bool = True) -> str:
        return self._run_capture_command(["wpctl", *list(args)], timeout_s=timeout_s, check=check)

    def _bt_audio_current_device_mac(self) -> str:
        if not hasattr(self, "cmb_bt_audio_devices"):
            return ""
        data = self.cmb_bt_audio_devices.currentData()
        return str(data or "").strip()

    def _bt_audio_set_indicator(self, connected: bool, state_text: str, detail_text: str, is_error: bool = False):
        if not hasattr(self, "lbl_bt_audio_state"):
            return

        if connected:
            frame_bg = "#14261a"
            frame_border = "#2e7d32"
            dot_color = "#39d353"
            state_color = "#d9ffe2"
            detail_color = "#b9d6c1"
        elif is_error:
            frame_bg = "#2c1717"
            frame_border = "#c62828"
            dot_color = "#ef5350"
            state_color = "#ffd9d9"
            detail_color = "#f3b4b4"
        else:
            frame_bg = "#2a2013"
            frame_border = "#ef6c00"
            dot_color = "#ff9800"
            state_color = "#ffe5c2"
            detail_color = "#e9c9a2"

        self.bt_audio_status_card.setStyleSheet(
            f"""
            QFrame#btAudioStatusCard {{
                background-color: {frame_bg};
                border: 1px solid {frame_border};
                border-radius: {self.s(12)}px;
            }}
            """
        )
        dot_size = self.s(14)
        self.lbl_bt_audio_dot.setStyleSheet(
            f"background-color:{dot_color}; border-radius:{max(1, dot_size // 2)}px;"
        )
        self.lbl_bt_audio_state.setText(str(state_text or "").strip())
        self.lbl_bt_audio_state.setStyleSheet(
            f"font-size:{self.s(18)}px; font-weight:bold; color:{state_color};"
        )
        self.lbl_bt_audio_detail.setText(str(detail_text or "").strip())
        self.lbl_bt_audio_detail.setStyleSheet(
            f"font-size:{self.s(12)}px; color:{detail_color};"
        )

    def _bt_audio_fetch_device_info(self, mac: str) -> dict:
        info = {
            "mac": str(mac or "").strip(),
            "name": str(mac or "").strip(),
            "connected": False,
            "paired": False,
            "trusted": False,
            "audio_candidate": False,
        }
        if not info["mac"]:
            return info

        try:
            output = self._run_btctl(["info", info["mac"]], timeout_s=8.0, check=False)
        except Exception:
            return info

        for raw_line in str(output).splitlines():
            line = raw_line.strip()
            if line.startswith("Name:"):
                info["name"] = line.split(":", 1)[1].strip() or info["name"]
            elif line.startswith("Alias:") and info["name"] == info["mac"]:
                info["name"] = line.split(":", 1)[1].strip() or info["name"]
            elif line.startswith("Connected:"):
                info["connected"] = line.split(":", 1)[1].strip().lower() == "yes"
            elif line.startswith("Paired:"):
                info["paired"] = line.split(":", 1)[1].strip().lower() == "yes"
            elif line.startswith("Trusted:"):
                info["trusted"] = line.split(":", 1)[1].strip().lower() == "yes"
            elif line.startswith("Icon:"):
                icon = line.split(":", 1)[1].strip().lower()
                if any(key in icon for key in ("audio", "headset", "headphones", "speaker")):
                    info["audio_candidate"] = True

            line_lc = line.lower()
            if any(
                key in line_lc
                for key in (
                    "audio sink",
                    "audio source",
                    "headset",
                    "handsfree",
                    "avrcp",
                    "a/v remote",
                )
            ):
                info["audio_candidate"] = True

        return info

    def _bt_audio_collect_known_devices(self):
        output = self._run_btctl(["devices"], timeout_s=8.0, check=False)
        devices = []
        for raw_line in str(output).splitlines():
            line = raw_line.strip()
            if not line.startswith("Device "):
                continue
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            mac = parts[1].strip()
            name = parts[2].strip() or mac
            info = self._bt_audio_fetch_device_info(mac)
            if info.get("name") == mac:
                info["name"] = name
            devices.append(info)
        return devices

    def _bt_audio_refresh_devices(self):
        if not hasattr(self, "cmb_bt_audio_devices"):
            return

        if not sys.platform.startswith("linux"):
            self.cmb_bt_audio_devices.clear()
            self.cmb_bt_audio_devices.addItem("Device not available", "")
            self.btn_bt_audio_refresh.setEnabled(False)
            self.btn_bt_audio_connect.setEnabled(False)
            self.btn_bt_audio_disconnect.setEnabled(False)
            self._bt_audio_set_indicator(
                False,
                "Bluetooth audio unavailable",
                "Bluetooth is not supported on this device.",
                is_error=True,
            )
            return

        preserve_mac = self._bt_audio_current_device_mac()
        try:
            all_devices = self._bt_audio_collect_known_devices()
            audio_devices = [d for d in all_devices if d.get("audio_candidate")]
            if not audio_devices:
                audio_devices = all_devices
            self.bt_audio_devices = audio_devices
        except Exception as e:
            self.bt_audio_devices = []
            self.cmb_bt_audio_devices.clear()
            self.cmb_bt_audio_devices.addItem("Bluetooth unavailable", "")
            self.btn_bt_audio_connect.setEnabled(False)
            self.btn_bt_audio_disconnect.setEnabled(False)
            self._bt_audio_set_indicator(
                False,
                "Bluetooth audio unavailable",
                str(e),
                is_error=True,
            )
            return

        self.cmb_bt_audio_devices.blockSignals(True)
        self.cmb_bt_audio_devices.clear()
        selected_index = -1
        for idx, dev in enumerate(self.bt_audio_devices):
            name = str(dev.get("name") or dev.get("mac") or "Unknown").strip()
            mac = str(dev.get("mac") or "").strip()
            suffix = []
            if dev.get("connected"):
                suffix.append("connected")
            elif dev.get("paired"):
                suffix.append("paired")
            label = f"{name} [{mac}]"
            if suffix:
                label += f" ({', '.join(suffix)})"
            self.cmb_bt_audio_devices.addItem(label, mac)
            if preserve_mac and mac == preserve_mac:
                selected_index = idx

        if self.cmb_bt_audio_devices.count() <= 0:
            self.cmb_bt_audio_devices.addItem("No known Bluetooth device", "")
        elif selected_index >= 0:
            self.cmb_bt_audio_devices.setCurrentIndex(selected_index)
        self.cmb_bt_audio_devices.blockSignals(False)

        connected = next((d for d in self.bt_audio_devices if d.get("connected")), None)
        if connected is not None:
            self.bt_audio_connected_mac = str(connected.get("mac") or "")
            self.bt_audio_connected_name = str(connected.get("name") or self.bt_audio_connected_mac)
            detail = "Connected. Press Connect Headset to route tone output to this device."
            if self.bt_audio_sink_id is not None:
                detail = f"Connected and routed to sink {self.bt_audio_sink_id}."
            self._bt_audio_set_indicator(
                True,
                "Headset connected",
                detail,
            )
        elif self.bt_audio_devices:
            self.bt_audio_connected_mac = ""
            self.bt_audio_connected_name = ""
            self._bt_audio_set_indicator(
                False,
                "Headset not connected",
                "Select a known Bluetooth audio device and press Connect.",
            )
        else:
            self.bt_audio_connected_mac = ""
            self.bt_audio_connected_name = ""
            self._bt_audio_set_indicator(
                False,
                "No headset found",
                "Pair the headset in Raspberry Pi OS first, then press Refresh.",
            )

        has_selection = bool(self._bt_audio_current_device_mac())
        self.btn_bt_audio_refresh.setEnabled(True)
        self.btn_bt_audio_connect.setEnabled(has_selection)
        self.btn_bt_audio_disconnect.setEnabled(
            bool(self.bt_audio_connected_mac or has_selection)
        )

    def _bt_audio_find_matching_sink(self, mac: str, name: str):
        status = self._run_wpctl(["status"], timeout_s=8.0)
        in_sinks = False
        sink_rows = []
        for raw_line in str(status).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("Sinks:"):
                in_sinks = True
                continue
            if not in_sinks:
                continue
            if stripped.startswith(
                (
                    "Sink endpoints:",
                    "Sources:",
                    "Source endpoints:",
                    "Streams:",
                    "Filters:",
                    "Clients:",
                    "Settings:",
                )
            ):
                break
            match = re.search(r"(\d+)\.\s+(.*)$", stripped)
            if match:
                sink_rows.append((int(match.group(1)), match.group(2).strip()))

        name_lc = str(name or "").strip().lower()
        mac_lc = str(mac or "").strip().lower()
        mac_us = mac_lc.replace(":", "_")
        mac_dash = mac_lc.replace(":", "-")
        best = None
        best_score = -1

        for sink_id, sink_label in sink_rows:
            try:
                inspect = self._run_wpctl(["inspect", str(sink_id)], timeout_s=6.0, check=False)
            except Exception:
                inspect = ""
            inspect_lc = str(inspect).lower()
            label_lc = str(sink_label).lower()
            score = 0
            if "bluez" in inspect_lc or "bluez" in label_lc or "bluetooth" in inspect_lc:
                score += 1
            if mac_lc and (mac_lc in inspect_lc or mac_us in inspect_lc or mac_dash in inspect_lc):
                score += 6
            if name_lc and name_lc in inspect_lc:
                score += 4
            if name_lc and name_lc in label_lc:
                score += 2
            if score > best_score:
                best = sink_id
                best_score = score

        if best_score <= 0:
            return None
        return best

    def _bt_audio_set_default_sink(self, sink_id: int):
        self._run_wpctl(["set-default", str(int(sink_id))], timeout_s=6.0)
        self.bt_audio_sink_id = int(sink_id)
        self._ble_log_line(f"[BT AUDIO] default sink set to {self.bt_audio_sink_id}")
        self._recreate_guidance_audio_backend()

    def _bt_audio_restore_local_sink(self):
        try:
            status = self._run_wpctl(["status"], timeout_s=8.0)
        except Exception:
            self._recreate_guidance_audio_backend()
            return

        in_sinks = False
        sink_rows = []
        for raw_line in str(status).splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("Sinks:"):
                in_sinks = True
                continue
            if not in_sinks:
                continue
            if stripped.startswith(
                (
                    "Sink endpoints:",
                    "Sources:",
                    "Source endpoints:",
                    "Streams:",
                    "Filters:",
                    "Clients:",
                    "Settings:",
                )
            ):
                break
            match = re.search(r"(\d+)\.\s+(.*)$", stripped)
            if match:
                sink_rows.append((int(match.group(1)), match.group(2).strip()))

        for sink_id, _sink_label in sink_rows:
            try:
                inspect = self._run_wpctl(["inspect", str(sink_id)], timeout_s=6.0, check=False)
            except Exception:
                inspect = ""
            if "bluez" in str(inspect).lower():
                continue
            self._run_wpctl(["set-default", str(int(sink_id))], timeout_s=6.0, check=False)
            self.bt_audio_sink_id = int(sink_id)
            self._recreate_guidance_audio_backend()
            return

        self.bt_audio_sink_id = None
        self._recreate_guidance_audio_backend()

    def _bt_audio_route_output_for_device(self, mac: str, name: str, retries: int = 5) -> bool:
        for _ in range(max(1, int(retries))):
            sink_id = self._bt_audio_find_matching_sink(mac, name)
            if sink_id is not None:
                self._bt_audio_set_default_sink(sink_id)
                return True
            QtWidgets.QApplication.processEvents()
            time.sleep(0.5)
        return False

    def _bt_audio_connect_selected(self):
        mac = self._bt_audio_current_device_mac()
        if not mac:
            self._bt_audio_set_indicator(
                False,
                "No headset selected",
                "Choose a Bluetooth headset first.",
                is_error=True,
            )
            return

        name = self.cmb_bt_audio_devices.currentText().split("[", 1)[0].strip() or mac
        self._bt_audio_set_indicator(
            False,
            "Connecting headset",
            f"Connecting to {name}...",
        )
        QtWidgets.QApplication.processEvents()

        try:
            self._run_btctl(["trust", mac], timeout_s=8.0, check=False)
            connect_output = self._run_btctl(["connect", mac], timeout_s=20.0, check=True)
            self._ble_log_line(f"[BT AUDIO] connect {mac}: {connect_output}")
            routed = self._bt_audio_route_output_for_device(mac, name)
            self._bt_audio_refresh_devices()
            if routed:
                self._bt_audio_set_indicator(
                    True,
                    "Headset connected",
                    f"Connected to {name}. Tone output routed to Bluetooth headset.",
                )
            else:
                self._bt_audio_set_indicator(
                    True,
                    "Headset connected",
                    f"Connected to {name}, but PipeWire sink was not found yet.",
                )
        except Exception as e:
            self._bt_audio_set_indicator(
                False,
                "Headset connect failed",
                str(e),
                is_error=True,
            )

    def _bt_audio_disconnect_selected(self):
        mac = self._bt_audio_current_device_mac() or str(self.bt_audio_connected_mac or "").strip()
        if not mac:
            self._bt_audio_set_indicator(
                False,
                "No headset selected",
                "Choose a Bluetooth headset first.",
                is_error=True,
            )
            return

        name = self.cmb_bt_audio_devices.currentText().split("[", 1)[0].strip() or mac
        self._bt_audio_set_indicator(
            False,
            "Disconnecting headset",
            f"Disconnecting {name}...",
        )
        QtWidgets.QApplication.processEvents()

        try:
            self._run_btctl(["disconnect", mac], timeout_s=12.0, check=False)
            self.bt_audio_connected_mac = ""
            self.bt_audio_connected_name = ""
            self.bt_audio_sink_id = None
            self._bt_audio_restore_local_sink()
            self._bt_audio_refresh_devices()
            self._bt_audio_set_indicator(
                False,
                "Headset disconnected",
                "Bluetooth headset audio disconnected. Tone output returned to local sink.",
            )
        except Exception as e:
            self._bt_audio_set_indicator(
                False,
                "Headset disconnect failed",
                str(e),
                is_error=True,
            )

    def _make_bt_icon(self, size_px: int) -> QtGui.QIcon:
        size_px = int(size_px) if size_px else 18
        size_px = 14 if size_px < 14 else size_px

        pm = QtGui.QPixmap(size_px, size_px)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        pen = QtGui.QPen(QtGui.QColor("#f0f0f0"))
        pen.setWidth(max(2, int(round(size_px / 10))))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        p.setPen(pen)

        cx = size_px / 2.0
        top = size_px * 0.12
        bot = size_px * 0.88
        mid = size_px * 0.50
        x1 = size_px * 0.25
        x2 = size_px * 0.75

        p.drawLine(QtCore.QPointF(cx, top), QtCore.QPointF(cx, bot))
        p.drawLine(QtCore.QPointF(cx, top), QtCore.QPointF(x2, mid))
        p.drawLine(QtCore.QPointF(x2, mid), QtCore.QPointF(cx, bot))
        p.drawLine(QtCore.QPointF(cx, top), QtCore.QPointF(x1, mid))
        p.drawLine(QtCore.QPointF(x1, mid), QtCore.QPointF(cx, bot))

        p.end()
        return QtGui.QIcon(pm)

    def _make_power_icon(self, size_px: int) -> QtGui.QIcon:
        size_px = int(size_px) if size_px else 18
        size_px = 14 if size_px < 14 else size_px

        pm = QtGui.QPixmap(size_px, size_px)
        pm.fill(QtCore.Qt.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        pen = QtGui.QPen(QtGui.QColor("#f0f0f0"))
        pen.setWidth(max(2, int(round(size_px / 10))))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)

        pad = size_px * 0.18
        rect = QtCore.QRectF(pad, pad, size_px - 2 * pad, size_px - 2 * pad)

        # Draw arc (almost a circle), leaving a gap at the top.
        p.drawArc(rect, int(40 * 16), int(280 * 16))

        # Draw the top vertical line.
        cx = size_px / 2.0
        y1 = pad * 0.6
        y2 = size_px * 0.52
        p.drawLine(QtCore.QPointF(cx, y1), QtCore.QPointF(cx, y2))

        p.end()
        return QtGui.QIcon(pm)

    def _confirm_power_action(self, title: str, text: str) -> bool:
        btn = QtWidgets.QMessageBox.question(
            self,
            title,
            text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return btn == QtWidgets.QMessageBox.Yes

    def _run_first_available(self, candidates):
        last_err = None
        for cmd in candidates:
            try:
                subprocess.run(cmd, check=True)
                return
            except FileNotFoundError as e:
                last_err = e
            except subprocess.CalledProcessError as e:
                last_err = e
            except Exception as e:
                last_err = e
        raise RuntimeError(str(last_err) if last_err else "Command failed")

    def _perform_reboot(self):
        if sys.platform.startswith("win"):
            raise RuntimeError("Reboot not supported.")
        try:
            self._on_stop()
        except Exception:
            pass
        self._run_first_available(
            [
                ["systemctl", "reboot"],
                ["reboot"],
                ["shutdown", "-r", "now"],
            ]
        )

    def _perform_shutdown(self):
        if sys.platform.startswith("win"):
            raise RuntimeError("Shutdown not supported.")
        try:
            self._on_stop()
        except Exception:
            pass
        self._run_first_available(
            [
                ["systemctl", "poweroff"],
                ["poweroff"],
                ["shutdown", "-h", "now"],
            ]
        )

    def _request_reboot(self):
        if sys.platform.startswith("win"):
            QtWidgets.QMessageBox.information(self, "Reboot", "Reboot not supported.")
            return
        if not self._confirm_power_action("Reboot", "Reboot perangkat sekarang?"):
            return
        try:
            self._perform_reboot()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Reboot failed", str(e))

    def _request_shutdown(self):
        if sys.platform.startswith("win"):
            QtWidgets.QMessageBox.information(self, "Shutdown", "Shutdown not supported.")
            return
        if not self._confirm_power_action("Shutdown", "Matikan perangkat sekarang?"):
            return
        try:
            self._perform_shutdown()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Shutdown failed", str(e))

    # ---------------------------------------------------------
    # BUILD UI
    # ---------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(self.s(4), self.s(4), self.s(4), self.s(4))
        root_layout.setSpacing(self.s(4))

        # ===== HEADER (title + battery) =====
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(self.s(8))

        title = QtWidgets.QLabel("Waltech Sigmavex")
        title.setStyleSheet(f"font-size:{self.s(14)}px; font-weight:bold; color:#dddddd;")
        header.addWidget(title, 1)

        self.lbl_battery = QtWidgets.QLabel("Battery: ...")
        self.lbl_battery.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_battery.setStyleSheet(f"font-size:{self.s(12)}px; color:#aaaaaa;")
        header.addWidget(self.lbl_battery, 0)

        root_layout.addLayout(header, 0)

        # ===== TAB WIDGET =====
        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs)

        # ---------- TAB 1: POWER METER ----------
        meter_page = QtWidgets.QWidget()

        # Portrait: keep footer buttons visible by scrolling the content area.
        meter_root_layout = QtWidgets.QVBoxLayout(meter_page)
        meter_root_layout.setContentsMargins(0, 0, 0, 0)
        meter_root_layout.setSpacing(0)

        meter_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(meter_content)
        main_layout.setContentsMargins(self.s(8), self.s(8), self.s(8), self.s(8))
        main_layout.setSpacing(self.s(8))

        meter_scroll = QtWidgets.QScrollArea()
        meter_scroll.setWidgetResizable(True)
        meter_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        meter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        meter_scroll.setWidget(meter_content)
        meter_root_layout.addWidget(meter_scroll, 1)

        # ===== KONTROL ATAS =====
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(self.s(10))
        grid.setVerticalSpacing(self.s(10) if self.is_portrait else self.s(6))
        if self.is_portrait:
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
        main_layout.addLayout(grid)

        # Mode
        lbl_mode = QtWidgets.QLabel("Mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            ChannelMode.MANUAL,
            ChannelMode.GSM900,
            ChannelMode.GSM1800,
            ChannelMode.LTE_B8_900,
            ChannelMode.LTE_B3_1800,
            ChannelMode.LTE_B1_2100,
            ChannelMode.LTE_B40_2300,
            ChannelMode.NR_N40_2300,
        ])

        # ARFCN / EARFCN / NR-ARFCN
        self.lbl_chan_desc = QtWidgets.QLabel("ARFCN:")
        self.chan_spin = QtWidgets.QSpinBox()
        self.chan_spin.setRange(0, 500000)
        self.chan_spin.setValue(1)

        # Gain
        lbl_gain = QtWidgets.QLabel("Gain:")
        self.gain_spin = QtWidgets.QSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setValue(int(DEFAULT_GAIN))
        self.gain_spin.setSuffix(" dB")

        # Freq Manual
        lbl_freq_manual = QtWidgets.QLabel("Freq Manual:")
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        # 1 decimal is enough for field use; avoids overly long values like 935.200000
        self.freq_spin.setDecimals(1)
        self.freq_spin.setRange(0.01, 6000.0)
        self.freq_spin.setSingleStep(0.1)
        self.freq_spin.setValue(935.2)
        self.freq_spin.setSuffix(" MHz")

        # Tombol SET
        self.btn_set = QtWidgets.QPushButton("SET PARA")
        self.btn_set.setMinimumHeight(self.s(42 if self.is_portrait else 34))
        self.btn_set.setMaximumWidth(self.s(180 if self.is_portrait else 170))
        self.btn_open_spectrum = QtWidgets.QPushButton("Spectrum")
        self.btn_open_spectrum.setMinimumHeight(self.s(42 if self.is_portrait else 34))
        self.btn_open_spectrum.setMaximumWidth(self.s(180 if self.is_portrait else 170))
        self.btn_open_spectrum.setVisible(bool(self.spectrum_mode_enabled))

        action_row = QtWidgets.QWidget()
        action_row_layout = QtWidgets.QHBoxLayout(action_row)
        action_row_layout.setContentsMargins(0, 0, 0, 0)
        action_row_layout.setSpacing(self.s(8))
        action_row_layout.addStretch(1)
        action_row_layout.addWidget(self.btn_set, 0)
        action_row_layout.addWidget(self.btn_open_spectrum, 0)
        action_row_layout.addStretch(1)

        # Taruh di grid:
        if self.is_portrait:
            grid.addWidget(lbl_mode,           0, 0)
            grid.addWidget(self.mode_combo,    0, 1)
            grid.addWidget(self.lbl_chan_desc, 1, 0)
            grid.addWidget(self.chan_spin,     1, 1)
            grid.addWidget(lbl_gain,           2, 0)
            grid.addWidget(self.gain_spin,     2, 1)
            grid.addWidget(lbl_freq_manual,    3, 0)
            grid.addWidget(self.freq_spin,     3, 1)
            grid.addWidget(action_row,         4, 0, 1, 2)
        else:
            grid.addWidget(lbl_mode,           0, 0)
            grid.addWidget(self.mode_combo,    0, 1)
            grid.addWidget(self.lbl_chan_desc, 0, 2)
            grid.addWidget(self.chan_spin,     0, 3)

            grid.addWidget(lbl_gain,        1, 0)
            grid.addWidget(self.gain_spin,  1, 1)
            grid.addWidget(lbl_freq_manual, 1, 2)
            grid.addWidget(self.freq_spin,  1, 3)

            grid.addWidget(action_row, 2, 0, 1, 4)

        # ===== INFO: Tuned & dBm =====
        self.lbl_tuned_freq = QtWidgets.QLabel("Tuned: -")
        self.lbl_tuned_freq.setAlignment(QtCore.Qt.AlignLeft)
        self.lbl_tuned_freq.setStyleSheet(f"font-size: {self.s(18)}px;")

        self.lbl_power = QtWidgets.QLabel("--.- dBm")
        self.lbl_power.setAlignment(QtCore.Qt.AlignRight)
        self.lbl_power.setStyleSheet(
            f"font-size: {self.s(28)}px; font-weight: bold; color: #b8b8b8;"
        )
        self.lbl_mode_note = QtWidgets.QLabel("")
        self.lbl_mode_note.setWordWrap(True)
        self.lbl_mode_note.setVisible(False)
        self.lbl_mode_note.setStyleSheet(
            f"font-size: {self.s(11)}px; color: #ffcc66;"
        )
        self.lbl_tdd_hint = QtWidgets.QLabel("")
        self.lbl_tdd_hint.setWordWrap(True)
        self.lbl_tdd_hint.setVisible(False)
        self.lbl_tdd_hint.setStyleSheet(
            f"font-size: {self.s(12)}px; color: #66ccff; font-weight: bold;"
        )

        if self.is_portrait:
            info_layout = QtWidgets.QVBoxLayout()
            info_layout.setSpacing(self.s(2))
            main_layout.addLayout(info_layout)

            self.lbl_tuned_freq.setAlignment(QtCore.Qt.AlignLeft)
            self.lbl_power.setAlignment(QtCore.Qt.AlignHCenter)

            info_layout.addWidget(self.lbl_tuned_freq)
            info_layout.addWidget(self.lbl_power)
        else:
            info_layout = QtWidgets.QHBoxLayout()
            main_layout.addLayout(info_layout)

            self.lbl_tuned_freq.setAlignment(QtCore.Qt.AlignLeft)
            self.lbl_power.setAlignment(QtCore.Qt.AlignRight)

            info_layout.addWidget(self.lbl_tuned_freq, 1)
            info_layout.addWidget(self.lbl_power, 1)

        main_layout.addWidget(self.lbl_mode_note)

        # ===== dBm BAR =====
        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setMinimum(int(DBM_MIN))
        self.level_bar.setMaximum(int(DBM_MAX))
        self.level_bar.setValue(int(DBM_MIN))
        self.level_bar.setTextVisible(True)
        self.level_bar.setAlignment(QtCore.Qt.AlignCenter)
        self._update_bar_style(color=(255, 0, 0), text="No signal")
        self.level_bar.setMinimumHeight(self.s(42))
        main_layout.addWidget(self.level_bar)

        # ===== History / Spectrum view =====
        self.spectrum = WaterfallWidget()
        self.spectrum_live = SpectrumTraceWidget()
        min_graph_h = self.s(260 if self.is_portrait else 160)
        self.spectrum.setMinimumHeight(min_graph_h)
        self.spectrum_live.setMinimumHeight(min_graph_h)
        self.spectrum.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.spectrum_live.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.spectrum_stack = QtWidgets.QStackedWidget()
        self.spectrum_stack.addWidget(self.spectrum)
        self.spectrum_stack.addWidget(self.spectrum_live)
        self.spectrum_stack.setCurrentWidget(self.spectrum)
        main_layout.addWidget(self.spectrum_stack, 1)

        # ===== Spectrum settings =====
        spectrum_cfg = QtWidgets.QWidget()
        spectrum_cfg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        if self.is_portrait:
            cfg = QtWidgets.QGridLayout(spectrum_cfg)
            cfg.setContentsMargins(0, 0, 0, 0)
            cfg.setHorizontalSpacing(self.s(8))
            cfg.setVerticalSpacing(self.s(4))
        else:
            cfg = QtWidgets.QHBoxLayout(spectrum_cfg)
            cfg.setContentsMargins(0, 0, 0, 0)
            cfg.setSpacing(self.s(10))

        lbl_fft = QtWidgets.QLabel("FFT")
        self.cmb_spec_fft = QtWidgets.QComboBox()
        self.cmb_spec_fft.addItems(["512", "1024", "2048", "4096"])
        self.cmb_spec_fft.setCurrentText(str(self.spectrum_fft_size))

        lbl_avg = QtWidgets.QLabel("Avg")
        self.spn_spec_avg = QtWidgets.QSpinBox()
        self.spn_spec_avg.setRange(1, 8)
        self.spn_spec_avg.setValue(int(self.spectrum_averages))

        lbl_rate = QtWidgets.QLabel("Rate")
        self.cmb_spec_rate = QtWidgets.QComboBox()
        self.cmb_spec_rate.addItems(["250 ms", "500 ms", "1000 ms"])
        self.cmb_spec_rate.setCurrentText(f"{int(self.spectrum_interval_s * 1000)} ms")

        self.chk_spec_maxhold = QtWidgets.QCheckBox("Max Hold")
        self.chk_spec_maxhold.setChecked(bool(self.spectrum_max_hold))

        small = f"font-size:{self.s(12)}px; color:#dddddd;"
        for w in (lbl_fft, lbl_avg, lbl_rate, self.chk_spec_maxhold):
            w.setStyleSheet(small)
        for w in (self.cmb_spec_fft, self.spn_spec_avg, self.cmb_spec_rate):
            w.setStyleSheet(f"font-size:{self.s(12)}px;")
            w.setMinimumHeight(self.s(28))

        if self.is_portrait:
            cfg.addWidget(lbl_fft, 0, 0)
            cfg.addWidget(self.cmb_spec_fft, 0, 1)
            cfg.addWidget(lbl_avg, 0, 2)
            cfg.addWidget(self.spn_spec_avg, 0, 3)
            cfg.addWidget(lbl_rate, 1, 0)
            cfg.addWidget(self.cmb_spec_rate, 1, 1)
            cfg.addWidget(self.chk_spec_maxhold, 1, 2, 1, 2)
            cfg.setColumnStretch(1, 1)
            cfg.setColumnStretch(3, 1)
        else:
            cfg.addWidget(lbl_fft)
            cfg.addWidget(self.cmb_spec_fft)
            cfg.addWidget(lbl_avg)
            cfg.addWidget(self.spn_spec_avg)
            cfg.addWidget(lbl_rate)
            cfg.addWidget(self.cmb_spec_rate)
            cfg.addWidget(self.chk_spec_maxhold)
            cfg.addStretch(1)

        main_layout.addWidget(spectrum_cfg, 0)

        # ===== Status + Start/Stop =====
        self.lbl_status = QtWidgets.QLabel("Status: Stopped")
        self.lbl_status.setAlignment(QtCore.Qt.AlignLeft)
        self.lbl_status.setStyleSheet(f"font-size: {self.s(14)}px;")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        for b in (self.btn_start, self.btn_stop):
            b.setMinimumHeight(self.s(60 if self.is_portrait else 50))
            b.setStyleSheet(f"font-size: {self.s(18)}px;")

        footer = QtWidgets.QWidget()
        if self.is_portrait:
            footer_layout = QtWidgets.QVBoxLayout(footer)
            footer_layout.setContentsMargins(self.s(8), self.s(6), self.s(8), self.s(8))
            footer_layout.setSpacing(self.s(6))

            buttons_h = QtWidgets.QHBoxLayout()
            buttons_h.setSpacing(self.s(8))

            footer_layout.addWidget(self.lbl_status)
            footer_layout.addLayout(buttons_h)
            buttons_h.addWidget(self.btn_start, 1)
            buttons_h.addWidget(self.btn_stop, 1)
        else:
            footer_layout = QtWidgets.QHBoxLayout(footer)
            footer_layout.setContentsMargins(self.s(8), self.s(6), self.s(8), self.s(8))
            footer_layout.setSpacing(self.s(8))

            footer_layout.addWidget(self.lbl_status, 2)
            footer_layout.addStretch(1)
            footer_layout.addWidget(self.btn_start, 1)
            footer_layout.addWidget(self.btn_stop, 1)

        meter_root_layout.addWidget(footer, 0)

        # Tambahkan tab meter ke QTabWidget
        self.tabs.addTab(meter_page, "DF Mode")

        # ---------- TAB 2: POWER (Reboot / Shutdown) ----------
        power_page = QtWidgets.QWidget()
        power_layout = QtWidgets.QVBoxLayout(power_page)
        power_layout.setContentsMargins(self.s(16), self.s(16), self.s(16), self.s(16))
        power_layout.setSpacing(self.s(12))

        power_title = QtWidgets.QLabel("Power")
        power_title.setAlignment(QtCore.Qt.AlignHCenter)
        power_title.setStyleSheet(f"font-size:{self.s(20)}px; font-weight:bold;")
        power_layout.addWidget(power_title)

        # power_hint = QtWidgets.QLabel("Gunakan menu ini untuk reboot / shutdown perangkat.")
        # power_hint.setAlignment(QtCore.Qt.AlignHCenter)
        # power_hint.setStyleSheet(f"font-size:{self.s(14)}px; color:#aaaaaa;")
        # power_layout.addWidget(power_hint)

        power_layout.addStretch(1)

        btn_reboot = QtWidgets.QPushButton("Reboot")
        btn_reboot.setMinimumHeight(self.s(64 if self.is_portrait else 54))
        btn_reboot.setStyleSheet(
            f"""
            QPushButton {{
                font-size:{self.s(20)}px;
                background-color:#2f2a1a;
                border: 1px solid #8a7a2a;
            }}
            QPushButton:pressed {{
                background-color:#3a3320;
            }}
            """
        )
        btn_reboot.clicked.connect(self._request_reboot)
        power_layout.addWidget(btn_reboot)

        btn_shutdown = QtWidgets.QPushButton("Shutdown")
        btn_shutdown.setMinimumHeight(self.s(64 if self.is_portrait else 54))
        btn_shutdown.setStyleSheet(
            f"""
            QPushButton {{
                font-size:{self.s(20)}px;
                background-color:#3a1f1f;
                border: 1px solid #aa4444;
            }}
            QPushButton:pressed {{
                background-color:#4a2727;
            }}
            """
        )
        btn_shutdown.clicked.connect(self._request_shutdown)
        power_layout.addWidget(btn_shutdown)

        power_layout.addStretch(2)

        self.tabs.setIconSize(QtCore.QSize(self.s(18), self.s(18)))
        power_tab_idx = self.tabs.addTab(power_page, "Power")
        self.tabs.setTabIcon(power_tab_idx, self._make_power_icon(self.s(18)))
        self.tabs.setTabToolTip(power_tab_idx, "Power")

        # ---------- TAB 3: BLE ----------
        ble_page = QtWidgets.QWidget()
        ble_layout = QtWidgets.QVBoxLayout(ble_page)
        ble_layout.setContentsMargins(self.s(16), self.s(16), self.s(16), self.s(16))
        ble_layout.setSpacing(self.s(10))

        ble_title = QtWidgets.QLabel("Bluetooth")
        ble_title.setAlignment(QtCore.Qt.AlignHCenter)
        ble_title.setStyleSheet(f"font-size:{self.s(18)}px; font-weight:bold;")
        ble_layout.addWidget(ble_title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(self.s(8))
        ble_layout.addLayout(row)

        ble_on = QtWidgets.QLabel("Bluetooth On")
        ble_on.setStyleSheet(f"font-size:{self.s(13)}px; color:#aaaaaa;")
        row.addWidget(ble_on, 0)

        row.addStretch(1)

        lbl_name = QtWidgets.QLabel("Name:")
        row.addWidget(lbl_name, 0)
        self.edt_ble_name = QtWidgets.QLineEdit()
        self.edt_ble_name.setText(str(self.ble_local_name))
        self.edt_ble_name.setReadOnly(True)
        self.edt_ble_name.setMaximumWidth(self.s(220))
        row.addWidget(self.edt_ble_name, 0)

        # hint = QtWidgets.QLabel(
        #     "Status below shows whether an Android phone is connected. BLE logs stay in terminal output."
        # )
        # hint.setWordWrap(True)
        # hint.setStyleSheet(f"font-size:{self.s(13)}px; color:#aaaaaa;")
        # ble_layout.addWidget(hint)

        self.ble_status_card = QtWidgets.QFrame()
        self.ble_status_card.setObjectName("bleStatusCard")
        ble_layout.addWidget(self.ble_status_card)

        ble_status_layout = QtWidgets.QHBoxLayout(self.ble_status_card)
        ble_status_layout.setContentsMargins(
            self.s(16), self.s(16), self.s(16), self.s(16)
        )
        ble_status_layout.setSpacing(self.s(12))

        self.lbl_ble_phone_dot = QtWidgets.QLabel()
        self.lbl_ble_phone_dot.setFixedSize(self.s(14), self.s(14))
        ble_status_layout.addWidget(
            self.lbl_ble_phone_dot, 0, QtCore.Qt.AlignTop
        )

        ble_status_text_layout = QtWidgets.QVBoxLayout()
        ble_status_text_layout.setSpacing(self.s(6))
        ble_status_layout.addLayout(ble_status_text_layout, 1)

        ble_status_title = QtWidgets.QLabel("Phone connection")
        ble_status_title.setStyleSheet(
            f"font-size:{self.s(13)}px; color:#b0b0b0; font-weight:bold;"
        )
        ble_status_text_layout.addWidget(ble_status_title)

        self.lbl_ble_phone_state = QtWidgets.QLabel("Phone not connected")
        ble_status_text_layout.addWidget(self.lbl_ble_phone_state)

        self.lbl_ble_phone_detail = QtWidgets.QLabel(
            "Waiting for Android device connection."
        )
        self.lbl_ble_phone_detail.setWordWrap(True)
        ble_status_text_layout.addWidget(self.lbl_ble_phone_detail)

        self.bt_audio_status_card = QtWidgets.QFrame()
        self.bt_audio_status_card.setObjectName("btAudioStatusCard")
        ble_layout.addWidget(self.bt_audio_status_card)

        bt_audio_layout = QtWidgets.QVBoxLayout(self.bt_audio_status_card)
        bt_audio_layout.setContentsMargins(
            self.s(16), self.s(16), self.s(16), self.s(16)
        )
        bt_audio_layout.setSpacing(self.s(10))

        bt_audio_head = QtWidgets.QHBoxLayout()
        bt_audio_head.setSpacing(self.s(12))
        bt_audio_layout.addLayout(bt_audio_head)

        self.lbl_bt_audio_dot = QtWidgets.QLabel()
        self.lbl_bt_audio_dot.setFixedSize(self.s(14), self.s(14))
        bt_audio_head.addWidget(self.lbl_bt_audio_dot, 0, QtCore.Qt.AlignTop)

        bt_audio_head_text = QtWidgets.QVBoxLayout()
        bt_audio_head_text.setSpacing(self.s(6))
        bt_audio_head.addLayout(bt_audio_head_text, 1)

        bt_audio_title = QtWidgets.QLabel("Headset audio")
        bt_audio_title.setStyleSheet(
            f"font-size:{self.s(13)}px; color:#b0b0b0; font-weight:bold;"
        )
        bt_audio_head_text.addWidget(bt_audio_title)

        self.lbl_bt_audio_state = QtWidgets.QLabel("Headset not connected")
        bt_audio_head_text.addWidget(self.lbl_bt_audio_state)

        self.lbl_bt_audio_detail = QtWidgets.QLabel(
            "Use a paired Bluetooth headset for tone audio output."
        )
        self.lbl_bt_audio_detail.setWordWrap(True)
        bt_audio_head_text.addWidget(self.lbl_bt_audio_detail)

        bt_audio_select_row = QtWidgets.QHBoxLayout()
        bt_audio_select_row.setSpacing(self.s(8))
        bt_audio_layout.addLayout(bt_audio_select_row)

        self.cmb_bt_audio_devices = QtWidgets.QComboBox()
        self.cmb_bt_audio_devices.setMinimumHeight(self.s(40))
        bt_audio_select_row.addWidget(self.cmb_bt_audio_devices, 1)

        self.btn_bt_audio_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_bt_audio_refresh.setMinimumHeight(self.s(40))
        bt_audio_select_row.addWidget(self.btn_bt_audio_refresh, 0)

        bt_audio_action_row = QtWidgets.QHBoxLayout()
        bt_audio_action_row.setSpacing(self.s(8))
        bt_audio_layout.addLayout(bt_audio_action_row)

        self.btn_bt_audio_connect = QtWidgets.QPushButton("Connect Headset")
        self.btn_bt_audio_disconnect = QtWidgets.QPushButton("Disconnect")
        for btn in (self.btn_bt_audio_connect, self.btn_bt_audio_disconnect):
            btn.setMinimumHeight(self.s(44))
        bt_audio_action_row.addWidget(self.btn_bt_audio_connect, 1)
        bt_audio_action_row.addWidget(self.btn_bt_audio_disconnect, 1)

        self.btn_bt_audio_refresh.clicked.connect(self._bt_audio_refresh_devices)
        self.btn_bt_audio_connect.clicked.connect(self._bt_audio_connect_selected)
        self.btn_bt_audio_disconnect.clicked.connect(self._bt_audio_disconnect_selected)

        ble_layout.addStretch(1)
        self._set_ble_connection_indicator(False, "Bluetooth is starting.")
        self._bt_audio_set_indicator(
            False,
            "Headset not connected",
            "Press Refresh to load known Bluetooth audio devices.",
        )

        ble_tab_idx = self.tabs.addTab(ble_page, "BLE")
        self.tabs.setTabIcon(ble_tab_idx, self._make_bt_icon(self.s(18)))

        # ---------- TAB 4: ABOUT ----------
        about_page = QtWidgets.QWidget()
        about_layout = QtWidgets.QVBoxLayout(about_page)
        about_layout.setContentsMargins(self.s(16), self.s(16), self.s(16), self.s(16))
        about_layout.setSpacing(self.s(8))

        about_label = QtWidgets.QLabel()
        about_label.setWordWrap(True)
        about_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        about_label.setText(
            f"""
            <table>
                <tr>
                    <td>
                        <img src="{str(IMG_DIR / 'wifi_icon2.ico')}" width="80" style="border-radius:10px;">
                    </td>
                    <td style="padding-left:12px;">
                        <h2>Waltech Sigmavex</h2>
                        <p><b>Software Version:</b> 0.1.0</p>
                        <p><b>Build date:</b> 29-11-25 (Prototype Version)</p>
                        <p><b>Portable Direction Finder</p>
                    </td>
                </tr>
            </table>

            <p><b>Hardware:</b></p>
            <ul>
                <li>Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.8GHz</li>
                <li>8GB LPDDR4-3200 SDRAM</li>
                <li>Wi-Fi IEEE 802.11ac 2.4 / 5 GHz + Bluetooth 5.0 BLE + Gigabit Ethernet</li>
                <li>RAT Support : GSM / LTE / 5G NSA / 5G NR</li>

            </ul>

            <p style="margin-top:14px; font-size:12px; color:#aaa;">
                &copy; 2025 Waltech Lab &mdash; @mhf
            </p>
            """
        )
        # On 480x800, About content often exceeds the viewport; keep it usable via scrolling.
        about_scroll = QtWidgets.QScrollArea()
        about_scroll.setWidgetResizable(True)
        about_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        about_scroll.setWidget(about_label)
        about_layout.addWidget(about_scroll)

        self.tabs.addTab(about_page, "About")

        # ===== Styling global =====
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #202020;
            }}
            QWidget {{
                background-color: #202020;
            }}
            QWidget#qt_scrollarea_viewport {{
                background-color: #202020;
            }}
            QScrollArea {{
                background-color: #202020;
            }}
            QLabel {{
                color: #f0f0f0;
                background-color: transparent;
            }}
            QPushButton {{
                padding: {self.s(6)}px {self.s(16)}px;
                color: #f0f0f0;
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: {self.s(6)}px;
            }}
            QPushButton:pressed {{
                background-color: #3a3a3a;
            }}
            QPushButton:disabled {{
                color: #777;
                background-color: #252525;
            }}
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
                font-size: {self.s(16)}px;
                min-height: {self.s(32)}px;
                color: #f0f0f0;
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: {self.s(6)}px;
                padding: {self.s(4)}px {self.s(8)}px;
            }}
            QComboBox::drop-down {{
                border-left: 1px solid #444;
            }}
            QComboBox QAbstractItemView {{
                background-color: #202020;
                color: #f0f0f0;
                selection-background-color: #00aa66;
                selection-color: #000000;
            }}
            QCheckBox {{
                color: #f0f0f0;
            }}
            QTabWidget::pane {{
                border: 1px solid #444;
                background-color: #202020;
            }}
            QTabBar::tab {{
                padding: {self.s(6)}px {self.s(14)}px;
                color: #d0d0d0;
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: {self.s(6)}px;
                border-top-right-radius: {self.s(6)}px;
                margin-right: {self.s(4)}px;
            }}
            QTabBar::tab:selected {{
                background-color: #202020;
                color: #ffffff;
            }}
        """)

        # ===== Signals =====
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        self.btn_set.clicked.connect(self._on_set)
        self.btn_open_spectrum.clicked.connect(self._toggle_spectrum_view)
        self.cmb_spec_fft.currentIndexChanged.connect(self._on_spectrum_cfg_changed)
        self.spn_spec_avg.valueChanged.connect(self._on_spectrum_cfg_changed)
        self.cmb_spec_rate.currentIndexChanged.connect(self._on_spectrum_cfg_changed)
        self.chk_spec_maxhold.toggled.connect(self._on_spectrum_cfg_changed)

        self._on_mode_changed()  # inisialisasi awal

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _update_bar_style(self, color=(255, 0, 0), text=""):
        r, g, b = color
        style = f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #202020;
                color: #ffffff;
                font-size: {self.s(18)}px;
            }}
            QProgressBar::chunk {{
                margin: 1px;
                border-radius: 3px;
                background-color: rgb({r}, {g}, {b});
            }}
        """
        self.level_bar.setStyleSheet(style)
        if text:
            self.level_bar.setFormat(text)

    def _get_active_tracker_info(self):
        info = self.spectrum_tracker_info
        if (
            not self.spectrum_mode_enabled
            or not hasattr(self, "spectrum_live")
            or
            self.spectrum_view_mode != "spectrum"
            or not isinstance(info, dict)
        ):
            return None
        tracked_dbm = float(info.get("tracked_dbm", np.nan))
        if not np.isfinite(tracked_dbm):
            return None
        return info

    def _update_spectrum_tracker(self, spectrum_dbm):
        if not self.spectrum_mode_enabled or not hasattr(self, "spectrum_live"):
            self.spectrum_tracker_info = None
            return None

        try:
            arr = np.asarray(spectrum_dbm, dtype=np.float32).reshape(-1)
        except Exception:
            self.spectrum_tracker_info = None
            if hasattr(self, "spectrum_live"):
                self.spectrum_live.set_tracker_info(None)
            return None

        if arr.size < 8 or np.all(np.isnan(arr)):
            self.spectrum_tracker_info = None
            if hasattr(self, "spectrum_live"):
                self.spectrum_live.set_tracker_info(None)
            return None

        valid = np.isfinite(arr)
        if not np.any(valid):
            self.spectrum_tracker_info = None
            if hasattr(self, "spectrum_live"):
                self.spectrum_live.set_tracker_info(None)
            return None

        if not np.all(valid):
            finite_vals = arr[valid]
            fill_value = float(np.mean(finite_vals)) if finite_vals.size else float(DBM_MIN)
            arr = np.where(valid, arr, fill_value)

        smooth = arr.astype(np.float64)
        if smooth.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= float(np.sum(kernel))
            smooth = np.convolve(smooth, kernel, mode="same")

        n_bins = int(smooth.size)
        edge_guard = max(6, int(n_bins * 0.06))
        search_lo = edge_guard
        search_hi = max(search_lo + 4, n_bins - edge_guard)

        global_peak_bin = int(search_lo + np.argmax(smooth[search_lo:search_hi]))
        chosen_bin = global_peak_bin

        prev = self.spectrum_tracker_info if isinstance(self.spectrum_tracker_info, dict) else None
        if prev and int(prev.get("fft_size", 0)) == n_bins:
            prev_bin = int(np.clip(int(prev.get("peak_bin", global_peak_bin)), 0, n_bins - 1))
            lock_half = max(12, int(n_bins * 0.05))
            local_lo = max(search_lo, prev_bin - lock_half)
            local_hi = min(search_hi, prev_bin + lock_half + 1)
            if local_hi > local_lo:
                local_peak_bin = int(local_lo + np.argmax(smooth[local_lo:local_hi]))
                if smooth[local_peak_bin] >= (smooth[global_peak_bin] - 4.0):
                    chosen_bin = local_peak_bin

        band_half = max(1, int(n_bins * 0.003))
        band_lo = max(0, chosen_bin - band_half)
        band_hi = min(n_bins, chosen_bin + band_half + 1)
        local_linear = np.power(10.0, arr[band_lo:band_hi].astype(np.float64) / 10.0)
        tracked_dbm = float(10.0 * np.log10(np.mean(local_linear) + 1e-15))
        peak_dbm = float(arr[chosen_bin])
        floor_dbm = float(np.percentile(arr.astype(np.float64), 20.0))

        sample_rate_hz = float(getattr(self.sdr, "sample_rate", 0.0)) if self.sdr is not None else 0.0
        if sample_rate_hz <= 1.0:
            _, sample_rate_hz = self._measurement_profile_for_mode()
        center_bin = (n_bins - 1) * 0.5
        freq_step_hz = sample_rate_hz / float(max(1, n_bins))
        if self.target_freq_hz is not None:
            tracked_freq_hz = float(self.target_freq_hz) + ((chosen_bin - center_bin) * freq_step_hz)
        else:
            tracked_freq_hz = None

        info = {
            "peak_bin": int(chosen_bin),
            "fft_size": int(n_bins),
            "tracked_dbm": float(tracked_dbm),
            "peak_dbm": float(peak_dbm),
            "floor_dbm": float(floor_dbm),
            "tracked_freq_hz": tracked_freq_hz,
            "ts": float(time.monotonic()),
        }
        self.spectrum_tracker_info = info
        if hasattr(self, "spectrum_live"):
            self.spectrum_live.set_tracker_info(info)
        return info

    def _toggle_spectrum_view(self):
        if not self.spectrum_mode_enabled:
            self.spectrum_view_mode = "history"
            self.spectrum_stack.setCurrentWidget(self.spectrum)
            self.btn_open_spectrum.setText("Spectrum")
            return

        if self.spectrum_view_mode == "history":
            self.spectrum_view_mode = "spectrum"
            self.spectrum_stack.setCurrentWidget(self.spectrum_live)
            self.btn_open_spectrum.setText("History")
            self.spectrum_live.set_max_hold_enabled(bool(self.spectrum_max_hold))
            if self._latest_spectrum_dbm is not None:
                self.spectrum_live.push_spectrum_dbm(self._latest_spectrum_dbm)
                self._update_spectrum_tracker(self._latest_spectrum_dbm)
        else:
            self.spectrum_view_mode = "history"
            self.spectrum_stack.setCurrentWidget(self.spectrum)
            self.btn_open_spectrum.setText("Spectrum")

    def _is_tdd_uplink_assist_active(self) -> bool:
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        return bool(
            self.tdd_uplink_assist_enabled
            and channel_mode_uses_shared_tdd_carrier(mode)
        )

    def _is_extra_slow_dbm_display_active(self) -> bool:
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        return mode == ChannelMode.LTE_B40_2300

    def _current_power_smooth_alpha(self) -> float:
        if self._is_tdd_uplink_assist_active():
            return float(TDD_UPLINK_ASSIST_SMOOTH_ALPHA)
        return float(POWER_SMOOTH_ALPHA)

    def _current_power_display_interval_s(self) -> float:
        if self._is_extra_slow_dbm_display_active():
            return 0.90
        if self._is_tdd_uplink_assist_active():
            return 0.55
        return 0.18

    def _current_power_display_alpha(self) -> float:
        if self._is_extra_slow_dbm_display_active():
            return 0.18
        if self._is_tdd_uplink_assist_active():
            return 0.24
        return 0.55

    def _update_display_power_dbm(self, source_dbm: float) -> float:
        now = time.monotonic()
        if np.isnan(self.display_power_dbm):
            self.display_power_dbm = float(source_dbm)
            self.last_power_display_ts = now
            return self.display_power_dbm

        if self._is_extra_slow_dbm_display_active():
            force_step_db = 4.0
        elif self._is_tdd_uplink_assist_active():
            force_step_db = 6.0
        else:
            force_step_db = 10.0
        interval_s = self._current_power_display_interval_s()
        should_refresh = (
            (now - self.last_power_display_ts) >= interval_s
            or abs(float(source_dbm) - self.display_power_dbm) >= force_step_db
        )

        if not should_refresh:
            return self.display_power_dbm

        alpha = self._current_power_display_alpha()
        if self._is_tdd_uplink_assist_active() and source_dbm < self.display_power_dbm:
            alpha = min(alpha, 0.18)

        self.display_power_dbm = (
            alpha * float(source_dbm)
            + (1.0 - alpha) * self.display_power_dbm
        )
        self.last_power_display_ts = now
        return self.display_power_dbm

    def _update_audio_power_dbm(self, source_dbm: float) -> float:
        source_dbm = float(source_dbm)
        if np.isnan(self.audio_power_dbm):
            self.audio_power_dbm = source_dbm
            return self.audio_power_dbm

        delta_db = source_dbm - self.audio_power_dbm
        if self._is_tdd_uplink_assist_active():
            rise_alpha = 0.82
            fall_alpha = 0.62
        else:
            rise_alpha = 0.90
            fall_alpha = 0.72

        if abs(delta_db) >= 10.0:
            alpha = 0.96
        elif delta_db >= 0.0:
            alpha = rise_alpha
        else:
            alpha = fall_alpha

        self.audio_power_dbm = (
            alpha * source_dbm
            + (1.0 - alpha) * self.audio_power_dbm
        )
        return self.audio_power_dbm

    def _measurement_profile_for_mode(self) -> tuple[float, float]:
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""

        gsm_modes = {
            ChannelMode.GSM900,
            ChannelMode.GSM1800,
        }
        fdd_tdd_modes = {
            ChannelMode.LTE_B8_900,
            ChannelMode.LTE_B3_1800,
            ChannelMode.LTE_B1_2100,
            ChannelMode.LTE_B40_2300,
            ChannelMode.NR_N40_2300,
        }

        if mode in gsm_modes:
            return (500e3, 2e6)
        if mode in fdd_tdd_modes:
            return (20e6, 25e6)
        return (float(DEFAULT_BANDWIDTH), float(DEFAULT_SAMPLE_RATE))

    def _apply_measurement_profile_to_sdr(self):
        if self.sdr is None:
            return
        target_bw, target_sr = self._measurement_profile_for_mode()
        try:
            if abs(float(getattr(self.sdr, "sample_rate", 0.0)) - target_sr) > 1.0:
                self.sdr.set_sample_rate(target_sr)
        except Exception as e:
            print("[SAMPLE RATE APPLY ERROR]", e)
        try:
            if abs(float(getattr(self.sdr, "bandwidth", 0.0)) - target_bw) > 1.0:
                self.sdr.set_bandwidth(target_bw)
        except Exception as e:
            print("[BANDWIDTH APPLY ERROR]", e)

    def _apply_measurement_profile_to_worker(self):
        if self.worker is None:
            return
        self.worker._cfg_lock.lock()
        try:
            self.worker.tdd_uplink_assist = self._is_tdd_uplink_assist_active()
        finally:
            self.worker._cfg_lock.unlock()

    def _refresh_mode_note(self):
        mode = self.mode_combo.currentText() if hasattr(self, "mode_combo") else ""
        note = channel_mode_measurement_note(mode)
        self.lbl_mode_note.setText(note)
        self.lbl_mode_note.setVisible(bool(note))

    def _reset_tdd_hint_display(self):
        self.display_power_dbm = np.nan
        self.last_power_display_ts = 0.0
        self.tdd_hint_peak_dbm = np.nan
        self.tdd_hint_steady_dbm = np.nan
        self.tdd_hint_burstiness_db = np.nan
        self.tdd_hint_display_state = ""
        self.tdd_hint_hold_until = 0.0
        if hasattr(self, "lbl_tdd_hint"):
            self.lbl_tdd_hint.clear()
            self.lbl_tdd_hint.setVisible(False)

    def _smooth_tdd_metric(self, previous_value: float, new_value: float, alpha: float) -> float:
        try:
            new_value = float(new_value)
        except Exception:
            return previous_value
        if np.isnan(new_value):
            return previous_value
        if np.isnan(previous_value):
            return new_value
        alpha = max(0.0, min(1.0, float(alpha)))
        return (alpha * new_value) + ((1.0 - alpha) * previous_value)

    def _tdd_hint_hold_duration_s(self, state: str) -> float:
        if state == "uplink_burst":
            return 1.1
        if state == "downlink_steady":
            return 0.8
        return 0.5

    def _choose_tdd_hint_state(self, candidate_state: str) -> str:
        state = str(candidate_state or "mixed")
        now = time.monotonic()

        if state == self.tdd_hint_display_state:
            self.tdd_hint_hold_until = now + self._tdd_hint_hold_duration_s(state)
            return state

        if (not self.tdd_hint_display_state) or (now >= self.tdd_hint_hold_until):
            self.tdd_hint_display_state = state
            self.tdd_hint_hold_until = now + self._tdd_hint_hold_duration_s(state)
            return state

        if state == "uplink_burst":
            self.tdd_hint_display_state = state
            self.tdd_hint_hold_until = now + self._tdd_hint_hold_duration_s(state)

        return self.tdd_hint_display_state

    def _render_tdd_hint(self, state: str, ul_peak_dbm: float, dl_steady_dbm: float, burstiness_db: float):
        return

    def _on_mode_changed(self):
        mode = self.mode_combo.currentText()
        if mode == ChannelMode.MANUAL:
            self.freq_spin.setEnabled(True)
            self.chan_spin.setEnabled(False)
            self.lbl_chan_desc.setEnabled(False)

        elif mode == ChannelMode.GSM900:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("ARFCN GSM900:")
            self.chan_spin.setRange(1, 124)
            self.chan_spin.setValue(2)

        elif mode == ChannelMode.GSM1800:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("ARFCN GSM1800:")
            self.chan_spin.setRange(512, 885)
            self.chan_spin.setValue(700)

        elif mode == ChannelMode.LTE_B8_900:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("EARFCN B8 900:")
            self.chan_spin.setRange(3450, 3799)
            self.chan_spin.setValue(3600)

        elif mode == ChannelMode.LTE_B3_1800:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("EARFCN B3 1800:")
            self.chan_spin.setRange(1200, 1949)
            self.chan_spin.setValue(1650)

        elif mode == ChannelMode.LTE_B1_2100:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("EARFCN B1 2100:")
            self.chan_spin.setRange(0, 599)
            self.chan_spin.setValue(300)

        elif mode == ChannelMode.LTE_B40_2300:
            b40_min, b40_max = lte_b40_earfcn_range()
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText(f"EARFCN B40 2300 ({b40_min}-{b40_max}):")
            self.chan_spin.setRange(b40_min, b40_max)
            self.chan_spin.setValue(39125)

        elif mode == ChannelMode.NR_N40_2300:
            self.freq_spin.setEnabled(False)
            self.chan_spin.setEnabled(True)
            self.lbl_chan_desc.setEnabled(True)
            self.lbl_chan_desc.setText("NR-ARFCN n40:")
            self.chan_spin.setRange(460000, 480000)
            self.chan_spin.setValue(470000)
        self._refresh_mode_note()
        self._apply_measurement_profile_to_sdr()
        self._apply_measurement_profile_to_worker()
        self.smoothed_power_dbm = np.nan
        self._reset_tdd_hint_display()
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    def _compute_freq_from_ui_hz(self) -> float:
        mode = self.mode_combo.currentText()
        if mode == ChannelMode.MANUAL:
            freq_mhz = self.freq_spin.value()
        else:
            freq_mhz = channel_to_freq_mhz(mode, self.chan_spin.value())
        return freq_mhz * 1e6

    def _on_set(self):
        """Ketika user tekan SET: simpan frekuensi target dan update label."""
        try:
            freq_hz = self._compute_freq_from_ui_hz()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error frekuensi/channel", str(e))
            return

        self.target_freq_hz = freq_hz
        freq_mhz = freq_hz / 1e6
        self.lbl_tuned_freq.setText(f"Tuned: {freq_mhz:.4f} MHz")

        if self.sdr is not None:
            try:
                self._apply_measurement_profile_to_sdr()
                self.sdr.tune(freq_hz)
                self._apply_measurement_profile_to_worker()
                self.smoothed_power_dbm = np.nan
                self._reset_tdd_hint_display()
                self.lbl_status.setText(
                    f"Status: Running (Gain {self.gain_spin.value()} dB)"
                )
                self.lbl_status.setStyleSheet(
                    f"font-size:{self.s(14)}px; color: lime; font-weight:bold;"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error tune SDR", str(e))
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    def _on_gain_changed(self, value: int):
        if self.sdr is not None:
            try:
                self.sdr.set_gain(value)
                self.lbl_status.setText(
                    f"Status: Running (Gain {value} dB)"
                )
                self.lbl_status.setStyleSheet(
                    f"font-size:{self.s(14)}px; color: lime; font-weight:bold;"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error set gain", str(e))
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    # ---------------------------------------------------------
    # Start / Stop
    # ---------------------------------------------------------
    def _on_start(self):
        # Kalau SDR belum ada / gagal init
        if self.sdr is None:
            if not self._prompt_sdr_reconnect():
                return
            self._init_sdr()
            if self.sdr is None:
                return  # tetap gagal setelah reconnect

        # --- SDR siap, start worker ---
        self.worker = PowerMeterWorker(self.sdr)
        # Apply current spectrum config to worker before thread starts.
        self.worker._cfg_lock.lock()
        try:
            self.worker.spectrum_fft_size = int(self.spectrum_fft_size)
            self.worker.spectrum_averages = int(self.spectrum_averages)
            self.worker.spectrum_interval_s = float(self.spectrum_interval_s)
            self.worker.tdd_uplink_assist = self._is_tdd_uplink_assist_active()
        finally:
            self.worker._cfg_lock.unlock()

        self._apply_measurement_profile_to_sdr()
        self.spectrum.set_max_hold_enabled(bool(self.spectrum_max_hold))
        if self.spectrum_mode_enabled:
            self.spectrum_live.set_max_hold_enabled(bool(self.spectrum_max_hold))
        else:
            self.spectrum_live.clear()
            self.spectrum_live.set_tracker_info(None)
        self.worker_thread = QtCore.QThread(self)
        self.worker.moveToThread(self.worker_thread)

        self.worker.powerUpdated.connect(self._on_power_updated)
        self.worker.spectrumUpdated.connect(self._on_spectrum_updated)
        self.worker.tddMetricsUpdated.connect(self._on_tdd_metrics_updated)
        self.worker.errorOccurred.connect(self._on_worker_error)
        self.worker_thread.started.connect(self.worker.start)
        self._reset_tdd_hint_display()
        self.measurement_running = True

        self.worker_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        freq_mhz = self.target_freq_hz / 1e6
        self.lbl_tuned_freq.setText(f"Tuned: {freq_mhz:.4f} MHz")

        self._start_guidance_tone()

        self.lbl_status.setText(
            f"Status: Running (Gain {self.gain_spin.value()} dB)"
        )
        self.lbl_status.setStyleSheet(
            f"font-size:{self.s(14)}px; color: lime; font-weight:bold;"
        )
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    def _on_stop(self):
        self.measurement_running = False

        if self.worker:
            self.worker.stop()
            QtCore.QThread.msleep(50)

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.worker = None
        self.worker_thread = None

        self._stop_guidance_tone()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.smoothed_power_dbm = np.nan
        self.audio_power_dbm = np.nan
        self.display_power_dbm = np.nan
        self.last_power_display_ts = 0.0
        self._latest_spectrum_dbm = None
        self.spectrum_tracker_info = None
        self.lbl_power.setText("--.- dBm")
        self.lbl_power.setStyleSheet(
            f"font-size: {self.s(28)}px; font-weight: bold; color: #b8b8b8;"
        )
        self.level_bar.setValue(int(DBM_MIN))
        self._update_bar_style(color=(255, 0, 0), text="No signal")
        self.lbl_status.setText("Status: Stopped")
        self.lbl_status.setStyleSheet(f"font-size:{self.s(14)}px; color: red;")
        if hasattr(self, "spectrum"):
            self.spectrum.clear()
        if hasattr(self, "spectrum_live"):
            self.spectrum_live.clear()
            self.spectrum_live.set_tracker_info(None)
        self.spectrum_view_mode = "history"
        if hasattr(self, "spectrum_stack"):
            self.spectrum_stack.setCurrentWidget(self.spectrum)
        if hasattr(self, "btn_open_spectrum"):
            self.btn_open_spectrum.setText("Spectrum")
        self._reset_tdd_hint_display()
        self._ble_spec_str = None
        self._reset_ble_status_cache()
        self._ble_push_status_if_enabled()

    # ---------------------------------------------------------
    # Tone / Audio handlers
    # ---------------------------------------------------------
    def _on_beep_status_changed(self):
        if self.beep is None:
            return
        status = self.beep.status()
        print("[TONE] fallback WAV status changed:", status)

    def _start_guidance_tone(self):
        self._last_tone_freq_hz = 280.0
        self._last_tone_volume = 0.0
        if self.tone_player.is_available():
            self.tone_player.start()
            self.tone_player.set_tone(280.0, 0.0)
            return

        if self.beep is not None and self.beep.status() == QtMultimedia.QSoundEffect.Ready:
            if not self.beep.isPlaying():
                self.beep.play()

    def _stop_guidance_tone(self):
        if self.tone_player.is_available():
            self.tone_player.stop()
        if self.beep is not None and self.beep.isPlaying():
            self.beep.stop()

    def _set_guidance_tone(self, frequency_hz: float, volume: float):
        if not self.measurement_running:
            self._stop_guidance_tone()
            return

        freq = float(np.clip(frequency_hz, 120.0, 1400.0))
        vol = float(np.clip(volume, 0.0, 1.0))
        self._last_tone_freq_hz = freq
        self._last_tone_volume = vol

        if self.tone_player.is_available():
            self.tone_player.set_tone(freq, vol)
            return

        if self.beep is None:
            return

        if self.beep.status() == QtMultimedia.QSoundEffect.Ready:
            if not self.beep.isPlaying():
                self.beep.play()
            self.beep.setVolume(vol)

    # ---------------------------------------------------------
    # Worker callback
    # ---------------------------------------------------------
    @QtCore.pyqtSlot(float)
    def _on_power_updated(self, power_dbm: float):
        if self.sdr is None or not self.measurement_running:
            return

        if np.isnan(power_dbm):
            self.smoothed_power_dbm = np.nan
            self.audio_power_dbm = np.nan
            self.display_power_dbm = np.nan
            self.last_power_display_ts = 0.0
            self.lbl_power.setText("No signal")
            self.lbl_power.setStyleSheet(
                f"font-size: {self.s(28)}px; font-weight: bold; color: #ff9f43;"
            )
            self.level_bar.setValue(int(DBM_MIN))
            self._update_bar_style(color=(255, 0, 0), text="No signal")
            self._set_guidance_tone(280.0, 0.0)
            self._reset_tdd_hint_display()
            if hasattr(self, "spectrum"):
                self.spectrum.clear()
            return

        # smoothing
        tracker_info = self._get_active_tracker_info()
        effective_dbm = float(power_dbm)
        if tracker_info is not None:
            effective_dbm = float(tracker_info.get("tracked_dbm", effective_dbm))

        if np.isnan(self.smoothed_power_dbm):
            self.smoothed_power_dbm = effective_dbm
        else:
            alpha = self._current_power_smooth_alpha()
            if self._is_tdd_uplink_assist_active():
                if effective_dbm >= self.smoothed_power_dbm:
                    alpha = min(0.28, max(alpha, 0.18))
                else:
                    alpha = max(0.06, alpha * 0.35)
            self.smoothed_power_dbm = (
                alpha * effective_dbm
                + (1.0 - alpha) * self.smoothed_power_dbm
            )

        disp_dbm = self._update_display_power_dbm(self.smoothed_power_dbm)
        audio_dbm = self._update_audio_power_dbm(effective_dbm)
        self.lbl_power.setText(f"{disp_dbm:0.1f} dBm")
        self.lbl_power.setStyleSheet(
            f"font-size: {self.s(28)}px; font-weight: bold; color: #00ff88;"
        )

        # clamp dan update bar
        p = disp_dbm
        if p < DBM_MIN:
            p = DBM_MIN
        if p > DBM_MAX:
            p = DBM_MAX

        self.level_bar.setMinimum(int(DBM_MIN))
        self.level_bar.setMaximum(int(DBM_MAX))
        self.level_bar.setValue(int(round(p)))
        if tracker_info is not None:
            self.level_bar.setFormat(f"Track {disp_dbm:0.1f} dBm")
        elif self._is_tdd_uplink_assist_active():
            self.level_bar.setFormat(f"UL hint {disp_dbm:0.1f} dBm")
        else:
            self.level_bar.setFormat(f"{disp_dbm:0.1f} dBm")

        # warna gradasi merah ke hijau
        ratio = (p - DBM_MIN) / (DBM_MAX - DBM_MIN)
        ratio = max(0.0, min(1.0, ratio))
        r = int((1.0 - ratio) * 255)
        g = int(ratio * 255)
        self._update_bar_style(color=(r, g, 0))
        if hasattr(self, "spectrum"):
            self.spectrum.set_level_dbm(disp_dbm)

        # Tone memakai jalur respons cepat agar gerak antena cepat langsung terasa.
        audio_p = audio_dbm
        if audio_p < DBM_MIN:
            audio_p = DBM_MIN
        if audio_p > DBM_MAX:
            audio_p = DBM_MAX
        audio_ratio = (audio_p - DBM_MIN) / (DBM_MAX - DBM_MIN)
        audio_ratio = max(0.0, min(1.0, audio_ratio))
        gate = (audio_p - (DBM_MIN + 4.0)) / 10.0
        gate = max(0.0, min(1.0, gate))

        if self._is_tdd_uplink_assist_active():
            min_freq = 360.0
            max_freq = 620.0
            min_vol = 0.025
            max_vol = 0.14
        else:
            min_freq = 380.0
            max_freq = 760.0
            min_vol = 0.030
            max_vol = 0.18

        tone_freq = min_freq + audio_ratio * (max_freq - min_freq)
        tone_vol = gate * (max_vol - audio_ratio * (max_vol - min_vol))
        self._set_guidance_tone(tone_freq, tone_vol)

    @QtCore.pyqtSlot(object)
    def _on_tdd_metrics_updated(self, metrics):
        if not self._is_tdd_uplink_assist_active():
            self._reset_tdd_hint_display()
            return

        if not isinstance(metrics, dict):
            return

        try:
            ul_peak_dbm = float(metrics.get("peak_dbm", np.nan))
            dl_steady_dbm = float(metrics.get("steady_dbm", np.nan))
            burstiness_db = float(metrics.get("burstiness_db", np.nan))
        except Exception:
            return

        if np.isnan(ul_peak_dbm) or np.isnan(dl_steady_dbm) or np.isnan(burstiness_db):
            return

        self.tdd_hint_peak_dbm = self._smooth_tdd_metric(
            self.tdd_hint_peak_dbm, ul_peak_dbm, 0.18
        )
        self.tdd_hint_steady_dbm = self._smooth_tdd_metric(
            self.tdd_hint_steady_dbm, dl_steady_dbm, 0.12
        )
        self.tdd_hint_burstiness_db = self._smooth_tdd_metric(
            self.tdd_hint_burstiness_db, burstiness_db, 0.20
        )

        state = self._choose_tdd_hint_state(
            str(metrics.get("classification", "mixed"))
        )
        self._render_tdd_hint(
            state=state,
            ul_peak_dbm=self.tdd_hint_peak_dbm,
            dl_steady_dbm=self.tdd_hint_steady_dbm,
            burstiness_db=self.tdd_hint_burstiness_db,
        )

    @QtCore.pyqtSlot(str)
    def _on_worker_error(self, msg: str):
        self._on_stop()
        QtWidgets.QMessageBox.critical(self, "Error worker", msg)

    @QtCore.pyqtSlot(object)
    def _on_spectrum_updated(self, spectrum_dbm):
        if self.sdr is None:
            return

        if spectrum_dbm is None:
            self._latest_spectrum_dbm = None
            self.spectrum_tracker_info = None
            if hasattr(self, "spectrum"):
                self.spectrum.clear()
            if hasattr(self, "spectrum_live"):
                self.spectrum_live.clear()
                self.spectrum_live.set_tracker_info(None)
            self._ble_spec_str = None
            self._ble_push_status_if_enabled()
            return

        try:
            self._latest_spectrum_dbm = np.asarray(spectrum_dbm, dtype=np.float32).reshape(-1).copy()
        except Exception:
            self._latest_spectrum_dbm = None

        if hasattr(self, "spectrum"):
            self.spectrum.push_spectrum_dbm(spectrum_dbm)
        if (
            self.spectrum_mode_enabled
            and hasattr(self, "spectrum_live")
            and self._latest_spectrum_dbm is not None
        ):
            self.spectrum_live.push_spectrum_dbm(self._latest_spectrum_dbm)
            self._update_spectrum_tracker(self._latest_spectrum_dbm)
        else:
            self.spectrum_tracker_info = None
            if hasattr(self, "spectrum_live"):
                self.spectrum_live.set_tracker_info(None)

        # Cache a small spectrum representation for BLE status (kept short to fit MTU).
        try:
            arr = np.asarray(spectrum_dbm, dtype=np.float32).reshape(-1)
            if arr.size >= 16:
                bins = 16
                idx = np.linspace(0, float(arr.size - 1), num=bins, dtype=np.float32)
                x_src = np.arange(arr.size, dtype=np.float32)
                ds = np.interp(idx, x_src, arr)
                q = np.clip(np.rint(ds), -140, 20).astype(np.int16)
                self._ble_spec_str = ";".join(str(int(v)) for v in q.tolist())
            else:
                self._ble_spec_str = None
        except Exception:
            self._ble_spec_str = None
        self._ble_push_status_if_enabled()

    def _on_spectrum_cfg_changed(self, *args):
        # Read UI values
        try:
            self.spectrum_fft_size = int(self.cmb_spec_fft.currentText())
        except Exception:
            self.spectrum_fft_size = 2048

        self.spectrum_averages = int(self.spn_spec_avg.value())

        rate_txt = self.cmb_spec_rate.currentText().strip().lower()
        if rate_txt.startswith("250"):
            self.spectrum_interval_s = 0.25
        elif rate_txt.startswith("500"):
            self.spectrum_interval_s = 0.50
        else:
            self.spectrum_interval_s = 1.00

        self.spectrum_max_hold = bool(self.chk_spec_maxhold.isChecked())
        self.spectrum.set_max_hold_enabled(self.spectrum_max_hold)
        if self.spectrum_mode_enabled:
            self.spectrum_live.set_max_hold_enabled(self.spectrum_max_hold)
        if self.spectrum_mode_enabled and self._latest_spectrum_dbm is not None:
            self.spectrum_live.push_spectrum_dbm(self._latest_spectrum_dbm)

        # Apply to worker if running
        if self.worker is not None:
            self.worker._cfg_lock.lock()
            try:
                self.worker.spectrum_fft_size = int(self.spectrum_fft_size)
                self.worker.spectrum_averages = int(self.spectrum_averages)
                self.worker.spectrum_interval_s = float(self.spectrum_interval_s)
            finally:
                self.worker._cfg_lock.unlock()

    def eventFilter(self, obj, event):
        # Popup keypad numerik untuk input (ARFCN, Gain, Freq Manual).
        if event.type() == QtCore.QEvent.MouseButtonPress:
            target = getattr(self, "_numeric_popup_targets", {}).get(obj)
            if target is not None:
                spinbox, kind = target
                if not spinbox.isEnabled():
                    return False

                current_val = spinbox.value()
                if kind == "float":
                    decimals = getattr(spinbox, "decimals", lambda: 1)()
                    initial_text = f"{float(current_val):.{decimals}f}"
                else:
                    initial_text = str(int(current_val))

                dlg = NumericKeypadDialog(self, initial_text)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    val = dlg.value()
                    if val is not None:
                        if kind == "int":
                            val = int(round(val))

                        # clamp ke range spinbox
                        if val < spinbox.minimum():
                            val = spinbox.minimum()
                        if val > spinbox.maximum():
                            val = spinbox.maximum()

                        spinbox.setValue(val)

                        # Setelah user ganti frekuensi, auto-update tuning.
                        if spinbox is self.freq_spin:
                            self._on_set()

                return True  # event sudah kita tangani

        return super().eventFilter(obj, event)


    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            QtCore.QThread.msleep(50)

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

        self._stop_guidance_tone()

        if self.sdr:
            self.sdr.close()
            print("[EXIT] SDR closed")

        if self.ble_server is not None:
            try:
                self.ble_server.stop()
            except Exception:
                pass
            self.ble_server = None
            self.ble_enabled = False

        if self._ina219_bus is not None:
            try:
                self._ina219_bus.close()
            except Exception:
                pass
            self._ina219_bus = None

        event.accept()


def run_app():
    app = QtWidgets.QApplication(sys.argv)

    # Lock UI layout to the 4.3in DSI target: landscape 800x480.
    # If the physical screen differs, we still scale from this base design.
    screen = app.primaryScreen()
    size = screen.size()
    base_w, base_h = 800.0, 480.0
    scale_w = size.width() / base_w
    scale_h = size.height() / base_h
    ui_scale = min(scale_w, scale_h)
    print(
        f"[UI] screen={size.width()}x{size.height()}, "
        f"target=800x480 landscape, ui_scale={ui_scale:.2f}"
    )

    win = MainWindow(scale=ui_scale, base_size=(int(base_w), int(base_h)), fullscreen=True)
    # showFullScreen sudah dipanggil di __init__ (fullscreen=True)
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
