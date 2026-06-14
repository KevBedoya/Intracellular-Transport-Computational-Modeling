from datetime import datetime


def return_timestamp():
    return datetime.now().strftime("%I-%M_%p_%m-%d-%Y")
