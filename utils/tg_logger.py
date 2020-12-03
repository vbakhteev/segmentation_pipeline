import json
import os
from typing import Tuple

import cv2
import numpy as np
import requests

from .visualization import metrics_to_image


def log_to_tg(model, experiment, checkpoint_callback):
    args, cfg = experiment["args"], experiment["cfg"]

    if not args.no_tg:
        tg_logger = experiment["tg_logger"]

        message = "{}\nExperiment {}".format(cfg.description, args.checkpoints_dir)
        if checkpoint_callback is not None:
            message += "\nBest {}: {:.4f}".format(
                checkpoint_callback.monitor,
                checkpoint_callback.best_model_score,
            )
        tg_logger.send_message(message)

        # TODO get validation loss
        metrics = model.logged_metrics
        if len(metrics):
            img = metrics_to_image(metrics)
            tg_logger.send_image(img)


class TelegramLogger:
    def __init__(self):
        self.credentials_file = "telegram_credentials.json"
        self.access_token, self.chat_id = self.get_token_and_chat_id()

    def send_message(self, message: str):
        message = message.replace("_", "\_")
        ping_url = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}".format(
            self.access_token, self.chat_id, message
        )

        response = requests.get(ping_url)

        return response

    def send_image(self, img: np.array):
        """img - RGB"""
        filename = "tmp.jpg"

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
        self.send_photo(filename)
        os.remove(filename)

    def send_photo(self, filepath):
        file_ = open(filepath, "rb")
        file_dict = {"photo": file_}
        ping_url = "https://api.telegram.org/bot{}/sendPhoto?chat_id={}".format(
            self.access_token, self.chat_id
        )
        response = requests.post(ping_url, files=file_dict)
        file_.close()

        return response

    def get_token_and_chat_id(self) -> Tuple[str, int]:
        if os.path.exists(self.credentials_file):
            return self.read_token_and_chat_id()

        try:
            access_token = input("Enter telegram access token: ")
            input("Send any message to bot in telegram and press enter.")
            chat_id = self.get_chat_id(access_token)
        except KeyError:
            print("Try again.")
            return self.get_token_and_chat_id()
        except Exception as e:
            print(e)
            raise

        self.write_token_and_chat_id(access_token, chat_id)

        return access_token, chat_id

    @staticmethod
    def get_chat_id(access_token) -> int:
        ping_url = "https://api.telegram.org/bot" + str(access_token) + "/getUpdates"
        response = requests.get(ping_url).json()
        chat_id = response["result"][0]["message"]["chat"]["id"]
        return chat_id

    def read_token_and_chat_id(self) -> Tuple[str, int]:
        with open(self.credentials_file, "r") as f:
            d = json.loads(f.read())
            access_token = d["access_token"]
            chat_id = d["chat_id"]
        return access_token, chat_id

    def write_token_and_chat_id(self, token: str, chat_id: int):
        d = dict(access_token=token, chat_id=chat_id)
        with open(self.credentials_file, "w") as f:
            json.dump(d, f)
