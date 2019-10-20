import os, requests
import tqdm


def download_weights(file_id, destination, chunk_size=32768):
    """
    Based on: https://stackoverflow.com/a/39225039
    """
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id}, stream=True)

    for key, token in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(url,
                params={'id': file_id, 'confirm': token}, stream=True)
            break
        
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        for chunk in tqdm.tqdm(response.iter_content(chunk_size)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk) 