import requests

WEBHOOK_URL = 'https://outlook.office.com/webhook/fb60d940-a0e9-41d4-bdb7-067fc3b9c85e@d72758a0-a446-4e0f-a0aa-4bf95a4a10e7/IncomingWebhook/bac3b3e70a1043c1aba2e7664fc8bedb/9e18086a-6e96-4260-84f0-f2e58833966a'


def send_success_message():
    msg = _build_message(['*PING* Your model is done! Fresh out the oven.'],
                         'https://media.giphy.com/media/3o6Zt8HWzhklRk36xy/giphy.gif')
    return _send_message(msg)


def send_failure_message():
    msg = _build_message(['❗ **Oh no, something went wrong** ❗'],
                         'https://media1.tenor.com/images/3a0c9909b542717ce9f651d07c4d4592/tenor.gif')
    return _send_message(msg)


def _send_message(obj):
    response = requests.post(WEBHOOK_URL, json=obj)
    if response.text == '1':
        return True
    return False


def _build_message(texts, gif=None):
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
