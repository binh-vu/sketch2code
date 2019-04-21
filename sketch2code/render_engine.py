import asyncio
from typing import List, Union

import imageio, zmq, ujson
from zmq.asyncio import Context
from pyppeteer import launch
from pyppeteer.browser import Browser
from pyppeteer.page import Page

from sketch2code.data_model import Tag, LinearizedTag


def patch_pyppeteer():
    import pyppeteer.connection
    original_method = pyppeteer.connection.websockets.client.connect

    def new_method(*args, **kwargs):
        kwargs['ping_interval'] = None
        kwargs['ping_timeout'] = None
        return original_method(*args, **kwargs)

    pyppeteer.connection.websockets.client.connect = new_method


patch_pyppeteer()
socket_template = "ipc:///tmp/sketch2code_render_engine_%d"


class RenderEngine:

    instance = None

    def __init__(self, browser: Browser, pages: List[Page], viewport_width: int,
                 viewport_height: int):
        self.browser = browser
        self.pages = pages
        self.n_pages = len(pages)

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def __del__(self):
        asyncio.get_event_loop().run_until_complete(self.browser.close())

    @staticmethod
    def get_instance(template_html: str,
                     n_cpu: int = 16,
                     viewport_width: int = 800,
                     viewport_height: int = 600,
                     headless: bool = True):
        if RenderEngine.instance is None:
            browser, pages = asyncio.get_event_loop().run_until_complete(
                RenderEngine.async_init(template_html, n_cpu, viewport_width, viewport_height,
                                        headless))

            RenderEngine.instance = RenderEngine(browser, pages, viewport_width, viewport_height)
        return RenderEngine.instance

    @staticmethod
    async def async_init(template_html: str, n_cpu: int, viewport_width: int, viewport_height: int,
                         headless: bool):
        browser = await launch(
            headless=headless,
            args=[
                # f'--window-size={viewport_width},{viewport_height}'
            ])
        pages = []

        for i in range(n_cpu):
            page = await browser.newPage()
            await page.setViewport({"width": viewport_width, "height": viewport_height})
            await page.setContent(template_html)
            pages.append(page)

        return browser, pages

    def render_pages(self,
                     tags: List[Union[Tag, LinearizedTag]],
                     format='jpeg',
                     full_page: bool = False):
        return asyncio.get_event_loop().run_until_complete(
            render_pages(self.pages, self.viewport_width, self.viewport_height, [x.to_body() for x in tags], format,
                         full_page))

    def render_page(self, tag: Union[Tag, LinearizedTag], format='jpeg', full_page: bool = False):
        return asyncio.get_event_loop().run_until_complete(
            render_page(self.pages[0], self.viewport_width, self.viewport_height, tag.to_body(),
                        format, full_page))


class RemoteRenderEngine:

    instance = None

    def __init__(self, template_html: str, format: str, full_page: bool, viewport_width: int,
                 viewport_height: int):
        self.full_page = full_page
        self.format = format
        self.template_html = template_html
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.context = zmq.Context()
        self.master_socket = self.context.socket(zmq.REQ)
        self.master_socket.connect(socket_template % 0)
        self.worker_sockets = []

        # init server and obtaining information
        self.master_socket.send_json({
            "cmd": "init",
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "template_html": self.template_html,
            "format": self.format,
            "full_page": self.full_page,
        })
        assert self.master_socket.recv() == b"Done"

        self.master_socket.send_json({"cmd": "info"})
        self.n_pages = self.master_socket.recv_json()['n_pages']
        for i in range(self.n_pages):
            s = self.context.socket(zmq.REQ)
            s.connect(socket_template % (i + 1))
            self.worker_sockets.append(s)

    def __del__(self):
        self.master_socket.close()
        for s in self.worker_sockets:
            s.close()
        self.context.term()

    @staticmethod
    def get_instance(template_html: str,
                     viewport_width: int = 800,
                     viewport_height: int = 600,
                     format: str = 'jpeg',
                     full_page: bool = False):
        if RemoteRenderEngine.instance is None:
            RemoteRenderEngine.instance = RemoteRenderEngine(template_html, format, full_page,
                                                             viewport_width, viewport_height)

        if template_html != RemoteRenderEngine.instance.template_html \
                or viewport_width != RemoteRenderEngine.instance.viewport_width \
                or viewport_height != RemoteRenderEngine.instance.viewport_height \
                or format != RemoteRenderEngine.instance.format \
                or full_page != RemoteRenderEngine.instance.full_page:
            raise Exception("Cannot re-create new instance of RemoteRenderEngine")

        return RemoteRenderEngine.instance

    @staticmethod
    def destroy():
        if RemoteRenderEngine.instance is not None:
            del RemoteRenderEngine.instance
            RemoteRenderEngine.instance = None

    def render_pages(self, tags: List[Union[Tag, LinearizedTag]]):
        self.master_socket.send_string(
            ujson.dumps({
                "cmd": "batch_render",
                "bodies": [tag.to_body() for tag in tags]
            }))

        return [imageio.imread(self.master_socket.recv()) for _ in range(len(tags))]

    def render_page(self, tag: Union[Tag, LinearizedTag]):
        self.worker_sockets[0].send_string(tag.to_body())
        return imageio.imread(self.worker_sockets[0].recv())


async def render_pages(pages: List[Page], vp_width: int, vp_height: int, bodies: List[str],
                       format: str, full_page: bool):
    event_loop = asyncio.get_event_loop()
    n_pages = len(pages)
    tasks = []
    results = []

    for i, body in enumerate(bodies):
        task = event_loop.create_task(
            render_page(pages[i % n_pages], vp_width, vp_height, body, format, full_page))
        tasks.append(task)

        if len(tasks) == n_pages:
            results += await asyncio.gather(*tasks)
            tasks = []

    if len(tasks) > 0:
        results += await asyncio.gather(*tasks)

    return results


async def render_page(page: Page, vp_width: int, vp_height: int, body: str, format: str,
                      full_page: bool) -> bytes:
    """Render a single page and return an image as bytes"""
    await page.evaluate('(x) => {document.body.innerHTML = x;}', body)
    args = {'type': format}
    if full_page:
        # doing this because full_page option doesn't work with current version of chromium
        # however, there is one caveat that function getBoundingClientRect doesn't take in to
        # account the margin, if we want to calculate it correctly, we have to compute the outer
        # height but it's going to be expensive
        bbox = await page.evaluate(
            "() => {bbox = document.body.getBoundingClientRect(); return [bbox.width, bbox.height]; }"
        )

        args['clip'] = {
            'x': 0,
            "y": 0,
            "width": max(vp_width, bbox[0]),
            "height": max(vp_height, bbox[1])
        }

    return await page.screenshot(args)


async def worker_exec(worker, page: Page, global_session: dict):
    while True:
        body = await worker.recv_string()
        result = await render_page(page, global_session['vp_width'], global_session['vp_height'],
                                   body, global_session['format'], global_session['full_page'])
        await worker.send(result)


async def master_exec(master, pages: List[Page], global_session: dict):
    print(f"[MASTER] >> Started")
    while True:
        msg = ujson.loads(await master.recv())
        if msg['cmd'] == 'init':
            vp_width = msg['viewport_width']
            vp_height = msg['viewport_height']
            for page in pages:
                await page.setViewport({"width": vp_width, "height": vp_height})
                await page.setContent(msg['template_html'])

            global_session['vp_width'] = vp_width
            global_session['vp_height'] = vp_height
            global_session['format'] = msg['format']
            global_session['full_page'] = msg['full_page']
            print(
                f"[MASTER] >> Reinitialize the render engine. Set viewport=({vp_width}, {vp_height})"
            )
            await master.send_string("Done")
        elif msg['cmd'] == 'info':
            await master.send_string(ujson.dumps({"n_pages": len(pages)}))
        elif msg['cmd'] == 'batch_render':
            results = await render_pages(pages, global_session['vp_width'],
                                         global_session['vp_height'], msg['bodies'],
                                         global_session['format'], global_session['full_page'])
            for i in range(len(results) - 1):
                await master.send(results[i], zmq.SNDMORE)
            await master.send(results[-1])
        else:
            raise Exception(f"Invalid message: {msg}")


async def start_server(headless: bool, n_cpu: int):
    global socket_template
    browser = await launch(
        headless=headless,
        args=[
            # f'--window-size={viewport_width},{viewport_height}'
        ])

    global_session = {}
    pages = []
    for i in range(n_cpu):
        page = await browser.newPage()
        pages.append(page)

    # start the server
    context = Context.instance()
    workers = []
    master = context.socket(zmq.REP)
    master.bind(socket_template % 0)

    try:

        for i in range(n_cpu):
            socket = context.socket(zmq.REP)
            socket.bind(socket_template % (i + 1))
            workers.append(socket)

        tasks = [
            asyncio.create_task(worker_exec(socket, page, global_session))
            for socket, page in zip(workers, pages)
        ]
        tasks.append(asyncio.create_task(master_exec(master, pages, global_session)))
        await asyncio.gather(*tasks)
    finally:
        await browser.close()
        for worker in workers:
            worker.close()
        master.close()
        context.term()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RemoteRenderEngine')
    parser.add_argument(
        '-m',
        '--headless',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Start Browser in headless mode')
    parser.add_argument('-c', '--concurrent', type=int, default=16, help='number of parallel jobs')
    parser.add_argument(
        '-d',
        '--dev',
        default=False,
        action="store_true",
        help='Run some test code (not start server)')
    args = parser.parse_args()

    if not args.dev:
        asyncio.run(start_server(args.headless == 'true', args.concurrent))
    else:
        import time, h5py, numpy as np
        from sketch2code.config import ROOT_DIR
        from sketch2code.data_model import *

        with open(ROOT_DIR / "datasets/toy/data.json", "r") as f:
            tags = [ToyTag.deserialize(o) for o in ujson.load(f)]

        start = time.time()
        render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), 480, 300)
        print(f"Start up take: {time.time() - start:.4f} seconds")

        start = time.time()
        results = render_engine.render_pages(tags)
        # results = [render_engine.render_page(tags[0])]
        print(
            f"Render a page take: {(time.time() - start) / len(results):.4f} seconds. ({len(results)} pages)"
        )

        start = time.time()
        print("max-width:", max(x.shape[0] for x in results), "max-height:",
              max(x.shape[1] for x in results))
        # results = np.asarray(results)
        # with h5py.File(ROOT_DIR / "datasets/pix2code/data.hdf5", "w") as f:
        #     dataset = f.create_dataset("images", results.shape, dtype='f')
        for i, example in enumerate(results[:10]):
            imageio.imwrite(f"./tmp/example_{i}.jpg", example)
        print(f"Misc process take: {time.time() - start:.4f} seconds")
        input("Enter to finish...")
