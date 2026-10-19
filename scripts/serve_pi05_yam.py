"""Serve the π0.5 spell policy on the Karma / MolmoAct2 YAM `/act` protocol.

    GET  /act -> health ({"norm_tag", "state_dim": 14, "num_cameras": 3, ...})
    POST /act -> json_numpy {top_cam, left_cam, right_cam (HxWx3 uint8 or 1-D
                 encoded JPEG), instruction, state (14,)} -> {"actions": (50, 14), "dt_ms"}

Karma runs `openpi-control infer --server http://HOST:PORT --instruction ...`.
Prediction only; this process never touches a robot. The TTT memory is not used
here: the protocol carries no executed-action feedback to commit.
"""
import argparse
import io
import logging
import threading
import time

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

from robo_ttt.pi05_adapter import Pi05Adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pi05.yam.server")

NORM_TAG = "yam_dual_pi05_spell"
STATE_DIM = 14
CAMERAS = (("top_cam", "top"), ("left_cam", "left_wrist"), ("right_cam", "right_wrist"))


def to_tensor(arr):
    a = np.asarray(arr)
    if a.ndim == 1:  # encoded JPEG/PNG bytes
        a = np.asarray(Image.open(io.BytesIO(a.astype(np.uint8).tobytes())).convert("RGB"))
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 or 1-D encoded, got {a.shape}")
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.uint8))).permute(2, 0, 1).float() / 255


class Server:
    def __init__(self, run, checkpoint, device):
        self.run, self.checkpoint, self.device = run, checkpoint, device
        self.adapter = Pi05Adapter.from_run(run, checkpoint=checkpoint, device=device)
        self.lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, payload):
        state = torch.as_tensor(np.asarray(payload["state"], dtype=np.float32)).reshape(-1)
        if state.shape != (STATE_DIM,) or not torch.isfinite(state).all():
            raise ValueError(f"expected finite state ({STATE_DIM},)")
        raw = {"observation.state": state[None], "task": [str(payload["instruction"])]}
        for key, name in CAMERAS:
            raw[f"observation.images.{name}"] = to_tensor(payload[key])[None]
        with self.lock:
            actions = self.adapter.predict(raw, use_memory=False)
        actions = actions[0].float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("nonfinite prediction")
        return actions


def error(status, message):
    return Response(json_numpy.dumps({"error": message}), status_code=status, media_type="application/json")


def build_app(server):
    app = FastAPI(title="pi05 YAM server")

    @app.get("/act")
    async def health():
        return JSONResponse({"status": "ok", "norm_tag": NORM_TAG, "run": server.run,
                             "checkpoint": server.checkpoint, "device": server.device,
                             "num_cameras": len(CAMERAS), "state_dim": STATE_DIM})

    @app.post("/act")
    async def act(request: Request):
        try:
            payload = json_numpy.loads((await request.body()).decode("utf-8"))
            for key in ("top_cam", "left_cam", "right_cam", "instruction", "state"):
                payload[key]
        except Exception as e:  # noqa: BLE001
            return error(400, f"bad request: {e}")
        t0 = time.perf_counter()
        try:
            actions = server.predict(payload)
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return error(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000
        return Response(json_numpy.dumps({"actions": actions, "dt_ms": dt_ms}), media_type="application/json")

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="/tmp/robo_ttt_pi05_spell_sft")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=4090)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()
    server = Server(args.run, args.checkpoint, args.device)
    if not args.no_warmup:
        img = np.zeros((180, 320, 3), np.uint8)
        t0 = time.perf_counter()
        server.predict({"top_cam": img, "left_cam": img, "right_cam": img, "instruction": "warmup",
                        "state": np.zeros(STATE_DIM, np.float32)})
        log.info("warmup OK (%.0f ms)", (time.perf_counter() - t0) * 1000)
    uvicorn.run(build_app(server), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
