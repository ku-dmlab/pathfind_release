import dataclasses
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from operator import itemgetter
from threading import Thread
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image
from selenium.common import WebDriverException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from env.common import Coordinate, PathfindID

def create_webdriver(headless: bool) -> Chrome:
    chrome_options = Options()
    # Disable unnecessary features
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--mute-audio")
    if headless:
        # Disable GPU hardware acceleration.
        chrome_options.add_argument("--disable-gpu")
        # Disable shared memory usage to prevent crashing in Docker container.
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Allow ChromeDriver to launch in root user.
        chrome_options.add_argument("--no-sandbox")
        # Launch Chrome in "new" headless mode;
        # See: https://developer.chrome.com/articles/new-headless/
        chrome_options.add_argument("--headless=new")
        
    return Chrome(
        options=chrome_options,
        service=ChromeService(ChromeDriverManager(driver_version="114.0.5735.90").install()),
    )
    
class PathfindService:
    """TODO Complete docstring.

    This class does three things:
    - Launch HTTP server that hosts `index.html` file on seperate thread.
    - Open the page in Selenium WebDriver.
    - Interact with Google Maps Javsacript API.
    """
    def __init__(
        self,
        api_key: str,
        headless: bool = True,
        server_address: tuple[str, int] = ("localhost", 29000),
        wait: float = 3.0,
        retry: int = 5,
    ):
        self.api_key = api_key
        self.URL: Final[str] = f"http://{server_address[0]}:{server_address[1]}"
        self.STATIC_API: Final[str] = (
            "https://maps.googleapis.com/maps/api/streetview?size=600x400&pano={0}&fov=80&heading={1}&pitch=0.0&key="
            + api_key
        )
        self.retry = retry

        self.headless = headless
        self._start_server(server_address)
        self.driver = create_webdriver(headless)
        self.driver.get(self.URL)
        self.wait = WebDriverWait(self.driver, wait)
        self._wait_for("sv")
        if not headless:
            self._execute_script("initializePanorama()")
            self._wait_for("panorama")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, traceback):
        self.close()
        
    def render(self, pf_id: PathfindID):
        if not self.headless:
            self._execute_script(f"panorama.setPano({pf_id.pano_id})")
            self._execute_script(f"panorama.setPov({{heading: {pf_id.heading}, pitch: 0.0}})")
            
    def capture(self, pf_id: PathfindID) -> npt.NDArray[np.float32]:
        url = self.STATIC_API.format(*dataclasses.astuple(pf_id))
        r = requests.get(url)
        r.raise_for_status()
        
        image = Image.open(BytesIO(r.content)).convert("RGB")
        image = np.array(image, dtype=np.float32)
        return image
    
    def move(self, pf_id: PathfindID, backward=False) -> PathfindID:
        try:
            data = self._get_panorama_data(pf_id.pano_id)
            diffs = []
            for link in data["links"]:
                # Compute difference, considering modulo 360.
                diff = abs(pf_id.heading - int(link["heading"]))
                diff = 360 - diff if diff > 180 else diff
                diffs.append((link["pano"], diff))
                
            data = self._get_panorama_data(min(diffs, key=itemgetter(1))[0])
            forward_key = PathfindID(data["pano"], pf_id.heading)
            
            data = self._get_panorama_data(max(diffs, key=itemgetter(1))[0])
            backward_key = PathfindID(data["pano"], pf_id.heading)
            
            return forward_key if backward else backward_key
            
        except ValueError as e:
            return pf_id
        
    def turn(self, pf_id: PathfindID, angle: int) -> PathfindID:
        return PathfindID(pf_id.pano_id, (pf_id.heading + angle) % 360)
    
    def get_id(self, coord: Coordinate) -> str:
        return self._execute_script(
            f"return sv.getPanorama({{location: {coord.location()}}}).then(processSVData)"
        )["pano"]
        
    def get_coordinate(self, pf_id: PathfindID) -> Coordinate:
        data = self._get_panorama_data(pf_id.pano_id)
        coord = Coordinate(data["lat"], data["lng"], pf_id.heading)
        return coord
    
    def close(self):
        self.driver.quit()
        
    def _start_server(self, server_address: tuple[str, int]):
        # TODO: Is it safe to discard Thread object?
        httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)
        thread = Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
    
    def _get_panorama_data(self, pano_id: str) -> dict:
        return self._execute_script(
            f"return sv.getPanorama({{pano: '{pano_id}'}}).then(processSVData)"
        )
        
    def _wait_for(self, target: str):
        self.wait.until(lambda _: self._execute_script(f"return typeof {target} !== 'undefined'"))
        
    def _execute_script(self, script: str) -> Any:
        last_exception = None
        for _ in range(self.retry):
            try:
                return self.driver.execute_script(script)
            except WebDriverException as e:
                self.driver.get(self.URL)
                self.wait.until(
                    lambda _: self.driver.execute_script("return typeof sv !== 'undefined'")
                )
                last_exception = e

        # self.driver.quit()
        raise TooManyRetriesError(
            f"Too many retries ({self.retry}) of {script}."
        ) from last_exception
        
class TooManyRetriesError(RuntimeError):
    pass