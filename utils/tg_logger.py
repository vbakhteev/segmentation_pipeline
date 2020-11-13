import json
import os
import tempfile
from typing import Tuple

import cv2
import numpy as np
import requests


class TelegramLogger:
    def __init__(self):
        self.credentials_file = "telegram_credentials.json"
        self.access_token, self.chat_id = self.get_token_and_chat_id()

    def send_message(self, message: str):
        ping_url = (
            "https://api.telegram.org/bot"
            + str(self.access_token)
            + "/sendMessage?chat_id="
            + str(self.chat_id)
            + "&parse_mode=Markdown&text="
            + message
        )
        response = requests.get(ping_url)

        return response

    def send_image(self, img: np.array):
        new_file, filename = tempfile.mkstemp()

        cv2.imwrite(filename, img)
        self.send_photo(filename)

        os.close(new_file)

    def send_photo(self, filepath):
        file_ = open(filepath, "rb")
        file_dict = {"photo": file_}
        ping_url = (
            "https://api.telegram.org/bot"
            + str(self.access_token)
            + "/sendPhoto?"
            + "chat_id="
            + str(self.chat_id)
        )
        response = requests.post(ping_url, files=file_dict)
        file_.close()

        return response

    def get_token_and_chat_id(self) -> Tuple[str, int]:
        if os.path.exists(self.credentials_file):
            return self.read_token_and_chat_id()

        access_token = input("Enter telegram access token: ")
        input("Send any message to telegram bot and press enter.")
        chat_id = self.get_chat_id()

        if not self.valid(access_token, chat_id):
            print("Try again.")
            return self.get_token_and_chat_id()

        self.write_token_and_chat_id(access_token, chat_id)

        return access_token, chat_id

    def valid(self, access_token: str, chat_id: int) -> bool:
        # TODO implement rules for validation
        return True

    def get_chat_id(self) -> int:
        ping_url = (
            "https://api.telegram.org/bot" + str(self.access_token) + "/getUpdates"
        )
        response = requests.get(ping_url).json()
        chat_id = response["result"][0]["message"]["chat"]["id"]
        return chat_id

    def read_token_and_chat_id(self) -> Tuple[str, int]:
        with open(self.credentials_file, "w") as f:
            d = json.loads(f.read())
            access_token = d["access_token"]
            user_id = d["user_id"]
        return access_token, user_id

    def write_token_and_chat_id(self, token: str, chat_id: int):
        d = dict(access_token=token, chat_id=chat_id)
        with open(self.credentials_file, "w") as f:
            json.dump(d, f)
