from twilio.rest import Client
import time
from sys import argv

from_num = "whatsapp:+00000000000"

def send_to_all_users(client, phoneNum, message):
    global from_num
    if "00000000000" not in from_num:
        client.messages.create(body=message, from_=from_num, to=phoneNum)
        time.sleep(3)


def send_msg(msg):
    ACCOUNT_SID = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    AUTH_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    numbers = ['whatsapp:+972000000000', 'whatsapp:+972000000000']
    for number in numbers:
        if "000000000" not in number:
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            send_to_all_users(client, number,msg)


if __name__ == "__main__":
    hour_counter = 0
    while True:
        time.sleep(3600)
        hour_counter += 1
        send_msg("השרת פועל %d שעות ללא הרצת המודל, אם הוא לא בשימוש נא לכבות אותו."%hour_counter)