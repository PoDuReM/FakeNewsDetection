from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vkr_prod.llm import saiga_llama3
from vkr_prod.ml import model
from vkr_prod.utils.vkr_prod_root import VKR_PROD_ROOT

from vkr_prod import inference

app = FastAPI()

search_generator = saiga_llama3.SearchGenerator()
classifier = model.NewsClassifier.from_config(VKR_PROD_ROOT / 'models/ru_bert.torch')
output_generator = saiga_llama3.OutputGenerator()


class RequestBody(BaseModel):
    title: str
    text: str


@app.post("/generate-answer")
def generate_answer(request_body: RequestBody):
    title = request_body.title
    text = request_body.text

    try:
        response = inference.think_about(title, text, search_generator, classifier,
                                         output_generator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
