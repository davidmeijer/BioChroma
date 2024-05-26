# -*- coding: utf-8 -*-

"""Script to parse data from the Hazardous Substances Data Bank (HSDB)"""

import json
import requests
import time
import typing as ty


def construct_url(page: int) -> str:
    """Construct the URL to request data from the HSDB database.
    
    :param page: The page number to request data from.
    :type page: int
    :return: The URL to request data from the HSDB database.
    :rtype: str
    """
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON/"
        "?source=Hazardous%20Substances%20Data%20Bank%20(HSDB)&heading_type=Compound&heading=UV%20Spectra"
        f"&page={page}"
        "&response_type=save"
        "&response_basename=PubChemAnnotations_Hazardous%20Substances%20Data%20Bank%20(HSDB)_heading%3DUV%20Spectra"
    )
    return url

def request_data(url: str) -> dict:
    """Request data from the HSDB database.
    
    :param url: The URL to request data from the HSDB database.
    :type url: str
    :return: The data from the HSDB database.
    :rtype: dict
    """
    response = requests.get(url)
    data = response.json()
    return data

def download_data() -> ty.List[dict]:
    """Download data from the HSDB database.
    
    :return: The data from the HSDB database.
    :rtype: ty.List[dict]
    """
    current_page = 1
    num_pages = float("inf")

    records = []
    while current_page <= num_pages:
        url = construct_url(current_page)
        data = request_data(url)
        
        num_pages = data["Annotations"]["TotalPages"]
        for record in data["Annotations"]["Annotation"]:
            records.append(record)

        current_page += 1
        time.sleep(1)
    
    return records

def main() -> None:
    """Main function to parse data from the HSDB database."""
    records = download_data()
    print(len(records))
    with open("data/hsdb.json", "w") as f:
        json.dump(records, f, indent=4)
    exit(0)


if __name__ == "__main__":
    main()
