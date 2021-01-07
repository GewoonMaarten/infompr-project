import requests

try:
    from utils.config import teams_webhook_url
except ImportError:
    print('_teams_webhook_url is not in config.json. This file cannot be used without it.\n')
    raise

def send_success_message():
    """ Send a success message to Teams """
    msg = _build_message(['*PING* Your model is done! Fresh out the oven.'],
                         'https://media.giphy.com/media/3o6Zt8HWzhklRk36xy/giphy.gif')
    return _send_message(msg)


def send_failure_message():
    """ Send a failuer message to Teams """
    msg = _build_message(['❗ **Oh no, something went wrong** ❗'],
                         'https://media1.tenor.com/images/3a0c9909b542717ce9f651d07c4d4592/tenor.gif')
    return _send_message(msg)


def _send_message(obj):
    """ Internal method """
    response = requests.post(teams_webhook_url, json=obj)
    if response.text == '1':
        return True
    return False


def _build_message(texts, gif=None):
    """ Internal method """
    base_dict = {
        '$schema': 'http://adaptivecards.io/schemas/adaptive-card.json',
        'type': 'MessageCard',
        'version': '1.0',
        'themeColor': 'FFA800',
        'summary': 'Notification',
        'sections': []
    }

    for text in texts:
        base_dict['sections'].append({
            'text': text
        })

    if gif:
        base_dict['sections'].append({
            'text': f'![If you don\'t see a gif here it means it was (re)moved.]({gif})'
        })

    return base_dict
