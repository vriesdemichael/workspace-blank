import os
import httpx

def download_model(url: str, destination_folder: str) -> None:
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()

        for file_name in response.json():
            file_url = f"{url}/{file_name}"
            file_response = client.get(file_url)
            file_response.raise_for_status()

            with open(os.path.join(destination_folder, file_name), "wb") as file:
                file.write(file_response.content)
