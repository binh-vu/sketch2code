import argparse
import time
import ujson
from typing import List, Dict, Tuple, Callable, Any, Optional

import asyncio

import numpy

from sketch2code.data_model import Tag
from pyppeteer import launch


def page2img(html: Tag):
    html.to_html()


async def start_server():
    browser = await launch(headless=True, args={
        "width": 800,
        "height": 600
    })
    page = await browser.newPage()
    await page.setViewport({"width": 800, "height": 600})

    with open("/Users/rook/workspace/GraduateStudy/CSCI-599-Deep-Learning-Spring-2019/Project/sketch2code/datasets/pix2code/data.json", 'r') as f:
        records = ujson.load(f)

    # set the layout first, then later we only need to update the body
    await page.setContent(Tag.deserialize(records[0]).to_html(indent=2))

    durations = []
    for i in range(10):
        start = time.time()
        body = Tag.deserialize(records[i]).to_body()
        await page.evaluate('(x) => {document.body.innerHTML = x;}', body)
        await page.screenshot({'fullPage': True, 'type': 'jpeg'})
        durations.append(time.time() - start)

    print(numpy.mean(durations))
    await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the render engine')
    # parser.add_argument('-m', '--mode', type=str, choices=['server', 'client'])

    # args = parser.parse_args()
    # if args.mode == 'server':
    asyncio.get_event_loop().run_until_complete(start_server())
    # elif args.mode == 'client'