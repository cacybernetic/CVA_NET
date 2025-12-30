import gc
import logging
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


class Monitor:

    def __init__(self) -> None:
        self._pbar: tqdm = None

    @property
    def pbar(self) -> tqdm:
        return self._pbar

    def create_pbar(self, total: int, desc: str="", leave: bool=False) -> None:
        self._pbar = tqdm(total=total, desc=desc, leave=leave)

    def print(self, text: str) -> None:
        if self._pbar is None:
            print(text)
        else:
            self._pbar.write(text)

    @staticmethod
    def log(text: str) -> None:
        LOGGER.info(text)

    def close_pbar(self) -> None:
        if self._pbar is None:
            return
        self._pbar.close()
        del self._pbar
        gc.collect()
        self._pbar = None
