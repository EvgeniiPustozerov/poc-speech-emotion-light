import requests

sound_file = "samples/crema_d_diarization/1.wav"


def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data


headers = {'authorization': "94f61366b3ac452a91380c1eebb6e2f3"}
response = requests.post('https://api.assemblyai.com/v2/upload',
                         headers=headers,
                         data=read_file(sound_file))

print(response.json())
