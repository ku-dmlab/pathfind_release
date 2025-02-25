from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np
from numpy import typing as npt
from strenum import StrEnum

Observation = npt.NDArray[np.float32]

@dataclasses.dataclass(frozen=True, init=False)
class Coordinate:
    lat: float
    lng: float
    heading: int

    def __init__(self, lat: float, lng: float, heading: int):
        object.__setattr__(self, "lat", round(lat, 6))
        object.__setattr__(self, "lng", round(lat, 6))
        object.__setattr__(self, "heading", heading)
        
    def __str__(self) -> str:
        return f"{self.lat}_{self.lng}_{self.heading}"

    def location(self) -> str:
        return f"{{lat: {self.lat}, lng: {self.lng}}}"
    
    def pov(self) -> str:
        return f"{{heading: {self.heading}, pitch: 0.0}}"
    
@dataclasses.dataclass(frozen=True)
class PathfindID:
    pano_id: str
    heading: int

    def __str__(self) -> str:
        return f"{self.pano_id}_{self.heading}"
    
class City(Enum):
    BOSTON = ("Boston", PathfindID("-Siaj8v6TsYnzPIiUaUjSg", 330), Coordinate(42.3600388, -71.0599952, 330))
    NEWYORK = ("New York", PathfindID("Uz5us5kujOzCLEPFvBM8vQ", 330), Coordinate(40.7128434, -74.0074149, 330))
    LONDON = ("London", PathfindID("KdF2w3nYQ0yVki8Ysq0IEA", 330), Coordinate(51.5075488, -0.1284081, 330))
    PARIS = ("Paris", PathfindID("N2v5eyk55zGumwVoQPDaqQ", 330), Coordinate(48.8571858, 2.3528322, 330), )
    LOSANGELES = ("Los Angeles", PathfindID("cJVDmkdYfDT9JjGzFJiWyw", 330), Coordinate(34.0522021, -118.2434529, 330))
    
    def __init__(self, name: str, pf_id: PathfindID, coord: Coordinate):
        self.city_name = name
        self.pf_id = pf_id
        self.coord = coord

class TargetText(StrEnum):
    ONE = "Two Traffic lights, one is green and the other one is yellow" # 
    TWO = "A man riding a bicycle is waiting in front of the streetlight"
    THREE = "A person on the bicycle is waiting for the traffic light" #
    FOUR = "Many cars on the road are stuck in traffic congestion"
    FIVE = "People are crossing the sidewalk." #
    SIX = "street lamps and parked cars are on the street"
    SEVEN = "There is a park, and it is surrounded by roads and buildings."
    EIGHT = "The scene where the car is about to cut in"
    NINE = "see a building on partial construction" #
    TEN = "A police car parked in front of a castle of marmor colours" #
    ELEVEN = "A telephone box in front of a bus stop with lots of people"
    TWELVE = "There is only one drive way between two buildings."
    THIRTEEN = "No cars are going forward thought it's green light."