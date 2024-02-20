import io

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model.inference_model import InferenceModel
import uvicorn

model = None
app = FastAPI()


class TextRequest(BaseModel):
    text: str


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model
    model = InferenceModel()


@app.post("/generate-file")
def generate_image_file(prompt: TextRequest):
    image = model.generate(prompt.text, 1)
    image[0].save("image.png")
    return FileResponse("image.png")


@app.post("/generate-stream")
def generate_image_stream(prompt: TextRequest):
    image = model.generate(prompt.text, 1)
    memory_stream = io.BytesIO()
    image[0].save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


app.mount("/", StaticFiles(directory="src/front/", html=True), name="site")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", reload=True, port=80)
