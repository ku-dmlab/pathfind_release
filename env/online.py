"""Gymnasium environment for Google StreetView.


The agent captures a 600x400-sized RGB image of the current panorama as its observation.

The agent can interact with the environment using five actions:
- (0) Move forward: navigate to the linked panorama with the closest heading.
- (1) Move backward: navigate to the linked panorama with the farthest heading.
- (2) Turn left: rotate the heading counterclockwise by 90 degree.
- (3) Turn right: rotate the heading clockwise by 90 degree.
- (4) Take picture: save a 600x400-sized RGB image from the current panorama.

The agent receives a reward based on the similarity between text and observation,
computed by a pre-trained CLIP model, only when it takes a picture. The other actions
return a reward of zero.

An episode ends either (1) when the agent takes a picture or (2) when it reaches the 
time limit (default: 80).

This environment utilizes Selenium with headless Chrome to access the Google Maps 
Javascript API. Consequently, the following requirements are necessary:

- Google Chrome executable and its associated dependencies.
- Google Maps API key.

:param render_mode: The render mode of the environment.
:param text: Target text for the reward.
:param device: The device for the reward computation. "cuda", "cpu", and "auto" are
    available. If set to "auto", it will default to "cuda", and fallback to "cpu" if
    CUDA is not available.
"""

import dataclasses
from typing import Any

import clip
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from bokeh.io import export_png
from bokeh.models import ColorBar, ColumnDataSource, GMapOptions, HoverTool
from bokeh.palettes import Plasma256 as palette
from bokeh.plotting import gmap
from bokeh.transform import linear_cmap
from gymnasium import spaces
from PIL import Image

from env.common import City, Coordinate, Observation, PathfindID, TargetText
from env.service import PathfindService, create_webdriver
from utils.logger import Logger

@dataclasses.dataclass(frozen=True)
class PathfindInfo:
    pf_id: PathfindID
    coord: Coordinate
    city: City
    episode_time_remaining: int
    city_time_remaining: int
    capture_remaining: int
    previous_reward: float

# TODO: Add a bill warning
class PathfindEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        api_key: str,
        service: PathfindService,
        logger: Logger,
        city_spec: list[list[str]],
        text_spec: list[str],
        capture_limit: int = 8,
        per_city_time_limit: int = 80,
        device: str = "auto",
        render_mode: str | None = None,
        fix_city_order: bool = False,
    ):
        self.api_key = api_key

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(512,))
        self.action_space = spaces.Discrete(5)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.city_spec = [[City[city_name] for city_name in city_names] for city_names in city_spec]
        self.texts = [TargetText[text] for text in text_spec]
        
        self.cities = self.city_spec[0]
        self.fix_city_order = fix_city_order
        self.pf_id = self.cities[0].pf_id
        self.city_idx = 0
        
        self.cnt_captures = 0
        self.max_captures = capture_limit

        self.service = service
        self.coor_freq = {}
        
        self.previous_reward = -1.0
        
        # Timestep
        self.city_t = 0
        self.episode_t = 0
        self.per_city_time_limit = per_city_time_limit
        
        self.logger = logger
        # TODO: Replace it with "with" context manager.
        self.logger_driver = create_webdriver(headless=True)
        self.num_ep = 0
        
        # Choose device for CLIP model.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Initialize the pre-trained CLIP model
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        start_id: PathfindID | None = None,
    ) -> tuple[Observation, PathfindInfo]:
        super().reset(seed=seed)
        
        if options is not None:
            if "cities" in options:
                self.cities: list[City] = options["cities"]
            if "text" in options:
                self.text: TargetText = options["text"]
        else:
            if self.fix_city_order:
                self.cities = self.city_spec[self.num_ep % len(self.city_spec)]
            else:
                cities_sublist = self.np_random.choice(self.city_spec)
                self.cities: list[City] = self.np_random.permutation(cities_sublist).tolist()
                
            self.text = self.np_random.choice(self.texts)
        
        self.total_time_limit = self.per_city_time_limit * len(self.cities)
        self.text_feature = self._compute_text_feature(self.text)
        if start_id is not None:
            self.pf_id = start_id
        else:
            self.pf_id = self.cities[0].pf_id
        self.city_idx = 0
        
        self.cnt_captures = 0
        
        self.num_ep += 1
        self.city_t = 0
        self.episode_t = 0
        self.previous_reward = 0.0
        
        return (self._get_observation(self.pf_id), self._get_info())
    
    def step(self, action: int) -> tuple[Observation, float, bool, bool, bool, PathfindInfo]:
        if action == 0:
            self.pf_id = self.service.move(self.pf_id)
        elif action == 1:
            self.pf_id = self.service.move(self.pf_id, backward=True)
        elif action == 2:
            self.pf_id = self.service.move(self.pf_id, 90)
        elif action == 3:
            self.pf_id = self.service.move(self.pf_id, -90)
        elif action == 4:
            self.cnt_captures += 1
        else:
            raise ValueError(f"{action} is not a valid action.")
        
        reward = self._get_reward(self.pf_id, action)
        
        self.city_t += 1
        self.episode_t += 1
        terminated = False
        city_changed = False
        if self.city_t == self.per_city_time_limit or self.cnt_captures == self.max_captures:
            self.city_t = 0
            self.cnt_captures = 0
            
            self.city_idx += 1
            city_changed = True
            if self.city_idx == len(self.cities):
                self.city_idx -= 1
                terminated = True
            self.pf_id = self.cities[self.city_idx].pf_id

        info = self._get_info()
        self.previous_reward = reward

        return (
            self._get_observation(self.pf_id),
            reward,
            terminated,
            False,
            city_changed,
            info,
        )
        
    def render(self) -> npt.NDArray[np.float32] | None:
        if self.render_mode == "rgb_array":
            return self.service.capture(self.pf_id)
        elif self.render_mode == "human":
            self.service.render(self.pf_id)
    
    def _get_observation(self, pf_id: PathfindID) -> Observation:
        image = self.service.capture(pf_id)
        feature = self._compute_feature(image)
        return feature.squeeze()
    
    def _get_reward(self, pf_id: PathfindID, action: int) -> float:
        if action != 4:
            return 0.0
        
        feature = self._get_observation(pf_id)
        reward = self._compute_rewards(feature)
        return reward
    
    def _get_info(self) -> PathfindInfo:
        return PathfindInfo(
            pf_id=self.pf_id,
            coord=self.service.get_coordinate(self.pf_id),
            city=self.cities[self.city_idx],
            episode_time_remaining=self.per_city_time_limit - self.episode_t,
            city_time_remaining=self.total_time_limit - self.city_t,
            capture_remaining=self.max_captures - self.cnt_captures,
            previous_reward=self.previous_reward,
        )
        
    @torch.no_grad()
    def _compute_feature(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # TODO: Use uniform type to minimize device transition / type conversion.
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_tensor: torch.Tensor = self.preprocess(image_pil)  # type: ignore
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return self.clip.encode_image(image_tensor).cpu().numpy()
    
    @torch.no_grad()
    def _compute_text_feature(self, text: str) -> npt.NDArray[np.float32]:
        tokens = clip.tokenize(text).to(self.device)
        text_feature = self.clip.encode_text(tokens)
        self.text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.text_feature
    
    @torch.no_grad()
    def _compute_rewards(self, observation: npt.NDArray[np.float32]) -> float:
        tensor_obs = torch.as_tensor(observation, device=self.device)
        normalized_obs = tensor_obs / tensor_obs.norm(dim=-1, keepdim=True)
        return (normalized_obs @ self.text_feature.T).item()
    
    def _dict_add_count_coord(self, dict, coord: Coordinate) -> dict[str, int]:
        key = (coord.lat, coord.lng)
        if key not in dict.keys():
            dict[key] = 1
        else:
            dict[key] += 1
        return dict
    
    def _build_df(self, dict) -> pd.DataFrame:
        df = pd.DataFrame(data=(dict.keys(), dict.values())).transpose()
        df.columns = ["coor", "count"]
        df[["lat", "lng"]] = pd.DataFrame(df["coor"].tolist(), index=df.index)
        return df
    
    def _plot(self, df, exp_df, lat, lng, city_name, zoom=15, map_type="roadmap"):
        gmap_options = GMapOptions(lat=lat, lng=lng, map_type=map_type, zoom=zoom)
        hover = HoverTool(tooltips=[("count", "@count")])
        bokeh_width, bokeh_height = 500, 400
        p = gmap(
            self.api_key,
            gmap_options,
            title="방문 Scatter Plot",
            width=bokeh_width,
            height=bokeh_height,
            tools=[hover],
        )
        source = ColumnDataSource(df)
        exp_source = ColumnDataSource(exp_df)
        mapper = linear_cmap("count", palette, 0, 99)
        center = p.circle("lng", "lat", radius=25, alpha=0.6, color=mapper, source=source)
        p.circle("lng", "lat", radius=25, alpha=1, color=mapper, source=exp_source)
        color_bar = ColorBar(color_mapper=mapper["transform"], location=(0, 0))
        p.add_layout(color_bar, "right")
        # png = export_png(p, filename=f"plot_{city_name}_zoom_{zoom}_zero_shot.png", webdriver=self.logger_driver)
        png = export_png(p, filename=f"plot_{city_name}_zoom_{zoom}_test.png", webdriver=self.logger_driver)
        return png
        
            

