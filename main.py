from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from chainlit.utils import mount_chainlit

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CSV Agent Explorer")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


mount_chainlit(app=app, target="cl_app.py", path="/chainlit")
