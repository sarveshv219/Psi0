import os
import sys
import tyro
import torch
import time
import numpy as np
import os.path as osp
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from PIL import Image
from typing import Union, Dict, Any, List
from base64 import b64decode, b64encode
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from torchvision.transforms import v2

from psi.deploy.helpers import *
from psi.config.config import LaunchConfig, ServerConfig
from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything
from psi.utils.overwatch import initialize_overwatch 

overwatch = initialize_overwatch(__name__)

class Server:
    
    def __init__(
        self, 
        policy:str, 
        run_dir: Path, 
        ckpt_step: int | str  = "latest", 
        device: str = "cuda:0", 
        enable_rtc: bool = False,
        action_exec_horizon: int | None = None
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
         
        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        # load launch config 
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config_.model_validate_json(conf)
        seed_everything(launch_config.seed or 42)

        from psi.models.psi0 import Psi0Model 
        self.model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=device)
        self.model.to(device)
        self.model.eval()

        self.maxmin:ActionStateTransform = launch_config.data.transform.field # type:ignore
        self.repack_transform:SimpleRepackTransform = launch_config.data.transform.repack # type:ignore
        self.model_transform:Psi0ModelTransform = launch_config.data.transform.model # type:ignore

        # Print number of total/trainable model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params*1e-6:.3f} Total", ctx_level=1)

        # self.previous_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32) # FIXME 
        # self.previous_height = np.array([0.74], dtype=np.float32)

        self.Da = launch_config.model.action_dim # type:ignore
        self.Tp = launch_config.model.action_chunk_size # type:ignore
        self.Ta = action_exec_horizon or launch_config.model.action_exec_horizon # type:ignore
        assert self.Ta <= self.Tp, "action_exec_horizon is too big"
        self.launch_config = launch_config
        self.count = 0
        
        self.enable_rtc = enable_rtc
        if enable_rtc:
            assert launch_config.model.rtc, "rtc is not supported for this model" #type:ignore
            self.rtc_max_delay = launch_config.model.max_delay  # type:ignore
            assert self.Tp - self.Ta <= self.rtc_max_delay, "action_exec_horizon is too big for the given rtc_max_delay and action_chunk_size"
            self.previous_action = None #np.zeros((self.Tp, self.Da), dtype=np.float32)
            overwatch.info(f"RTC enabled with max_delay={self.rtc_max_delay}, \n"
                           f"action_dim={self.Da}, \n"
                           f"action_chunk_size={self.Tp}, \n"
                           f"action_exec_horizon={self.Ta}")
        self.last_serve_time = time.monotonic()
        # Read here as well as in psi0.py's from_pretrained: that one selects the eager backend at
        # load, this one decides whether to spend a forward on the map. Both must agree, so both
        # read the same variable rather than one passing a flag to the other.
        self.want_attn = os.environ.get("PSI0_ATTN") == "1"
        if self.want_attn:
            overwatch.info("PSI0_ATTN=1 -- serving VLM text->image attention maps")


    @torch.inference_mode()
    def attention_map(self, imgs: List[Any], instruction: str):
        """VLM text->image attention -> ((R, h, w) float32, R token strings), or (None, None).

        **This is the only fused image+language quantity Qwen3-VL produces.** It is a causal
        decoder: the text tokens sit AFTER the image tokens, so text attends to image and never
        the reverse. Measured on this checkpoint, across the six commands with the image held
        fixed, `hidden_states[-1]` differs at positions 90-99 ONLY -- the 80 image tokens are
        bit-identical (max deviation 0.00e+00). So there is no ClearCLIP-style symmetric fusion
        where a language vector modulates the patch grid; the fusion lives entirely in the
        trailing text rows, and this is the map of it.

        One row per text token rather than a mean, because the question is which WORD looks where
        -- the commanded colour is a single token, and averaging it with "face is up." buries it.

        Costs one extra VLM forward per plan. The map does not change across denoising steps
        (same inputs), so it is computed once here rather than inside the sampling loop.

        The message construction MIRRORS Psi0Model.predict_action (psi0.py:1667-1686). The asserts
        below are what catch that mirror going stale.
        """
        try:
            from qwen_vl_utils import process_vision_info
            proc = self.model.vlm_processor
            messages = [[{"role": "user",
                          "content": [{"type": "image", "image": im} for im in imgs]
                                     + [{"type": "text", "text": instruction}]}]]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in messages]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = proc(text=texts, images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(self.device)
            out = self.model.vlm_model(**inputs, output_attentions=True, return_dict=True)
            if out.attentions is None:
                overwatch.warning("attentions is None -- the VLM was not loaded with eager "
                                  "attention. Restart the server with PSI0_ATTN=1.")
                return None, None

            ids = inputs["input_ids"][0]
            pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            cols = (ids == pad_id).nonzero().flatten()
            assert len(cols) > 0, "no <|image_pad|> in the prompt; the mirror of predict_action broke"
            assert int(cols[-1]) - int(cols[0]) + 1 == len(cols), "image tokens are not contiguous"
            thw = inputs["image_grid_thw"][0].tolist()
            ms = int(proc.image_processor.merge_size)
            h, w = thw[1] // ms, thw[2] // ms
            assert h * w == len(cols), f"grid {h}x{w} != {len(cols)} image tokens"

            rows = torch.arange(int(cols[-1]) + 1, ids.shape[0], device=ids.device)
            # Last layer, mean over heads. Attention is per-layer and per-head; the last layer is
            # the one whose output IS hidden_states[-1], i.e. the only thing the action expert
            # reads, so it is the layer whose routing is causally connected to the plan.
            a = out.attentions[-1][0].float().mean(0)          # (S, S)
            m = a[rows][:, cols].reshape(len(rows), h, w)      # (R, h, w)
            toks = [proc.tokenizer.decode([int(ids[i])]) for i in rows.tolist()]
            return m.cpu().numpy().astype(np.float32), toks
        except Exception:
            import traceback
            overwatch.warning("attention_map failed (non-fatal):\n" + traceback.format_exc())
            return None, None

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        # overwatch.info(f"Received request with payload: {payload}")
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
            
            overwatch.info(f"Instruction: {instruction}")
            overwatch.info(f"history_dict: {history_dict}")

            transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
            t = v2.Compose(transforms)
            # Transform ONCE and reuse for both the plan and the attention map, so the map is
            # guaranteed to describe the same pixels the plan was conditioned on.
            views = [t(Image.fromarray(img)) for img in image_dict.values()]

            states = torch.from_numpy(state_dict["states"].copy())

            if self.maxmin.normalize_state: # type:ignore
                # Guard pad_state_dim the way ActionStateTransform.__call__ itself does
                # (config/transform.py:59). Unguarded, a config that normalizes state without
                # padding it -- which is every SONIC-latent run, whose 32-d states already match
                # odim -- reaches `current_len >= None` inside pad_to_len and raises TypeError.
                # That lands in this method's except, so the server answers 200 with a status
                # string and NO action rather than failing; the client sees a missing key.
                s = states.numpy()
                if self.maxmin.pad_state_dim is not None: # type:ignore
                    s = pad_to_len(s, self.maxmin.pad_state_dim, dim=1)[0]
                states = torch.from_numpy(
                    self.maxmin.normalize_state_func(s)
                ).to(self.device)

            if not self.enable_rtc:
                raw_pred_actions = self.model.predict_action(
                    observations=[views], 
                    states=states.unsqueeze(0), # B, To, Ds
                    instructions=[instruction], # [Task] * B
                    num_inference_steps=10, 
                    traj2ds=None
                )
            else: # rtc
                current_time = time.monotonic()
                if self.previous_action is None or "reset" in history_dict: #  or (current_time - self.last_serve_time) > 30  #if idle more than 60s, reset previous action
                    overwatch.info("===Reset or first step, without condition===")
                    raw_pred_actions = self.model.predict_action(
                        observations=[views], 
                        states=states.unsqueeze(0), # B, To, Ds
                        instructions=[instruction], # [Task] * B
                        num_inference_steps=10, 
                        traj2ds=None
                    )
                else:
                    overwatch.info("RTC enabled, using RTC inference")
                    overwatch.info("Last chunk execution loop time: {:.2f}s ago".format(current_time - self.last_serve_time))
                    prev_actions = np.concatenate([
                        self.previous_action[None, self.Ta:, :], 
                        np.zeros((1, self.Ta, self.Da), dtype=np.float32)
                    ], axis=1) # (1, Tp, Da)
                    prev_actions = torch.from_numpy(prev_actions).to(self.device)

                    raw_pred_actions = self.model.predict_action_with_training_rtc_flow(
                        observations=[views], 
                        states=states.unsqueeze(0), # B, To, Ds
                        instructions=[instruction], # [Task] * B
                        num_inference_steps=10, 
                        traj2ds=None,
                        prev_actions=prev_actions,
                        inference_delay=(self.Tp - self.Ta), 
                        max_delay=self.rtc_max_delay
                    )

            raw_pred_actions = raw_pred_actions.reshape(-1, self.Da).cpu().numpy() # (Tp, Da)
            pred_actions = self.maxmin.denormalize(raw_pred_actions) # (Ta, Da)
            self.previous_action = raw_pred_actions.copy().astype(np.float32) # for rtc
            pred_actions = pred_actions[:self.Ta] # type:ignore
            overwatch.info(f"Return Action ({pred_actions.shape})") # : {pred_actions}

            attn, attn_tokens = (self.attention_map(views, instruction)
                                 if self.want_attn else (None, None))
            self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions, 0.0, attn=attn,  # type:ignore
                                       attn_tokens=attn_tokens)
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback
            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        overwatch.info(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            overwatch.warning(f"Server crashed, {e}")
        finally:
            overwatch.info("Server stopped.")
            exit(1)

def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Psi0")
    assert cfg.policy is not None, "which policy to serve?"
    server = Server(
        cfg.policy, 
        Path(cfg.run_dir), 
        cfg.ckpt_step, 
        cfg.device, 
        cfg.rtc,
        cfg.action_exec_horizon
    )
    
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv
    assert load_dotenv() 
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)

if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)